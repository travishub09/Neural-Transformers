"""Fresh all-neuron balanced groups and a small matched quality comparison."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import load_file
from pathpred.clustering import activation_features,balanced_kmeans,grouping_metrics,validate_permutation
from pathpred.collection import atomic_json,atomic_tensors,file_sha256
from pathpred.fresh_groups import original_to_stored_order,contribution_mask,contribution_gram
from pathpred.oracle_checks import quality_rows
from pathpred.reordering import load_clustered_qwen
from scripts.cluster_neurons import read_samples
from scripts.oracle_quality import aggregate,bootstrap_quality


def paired_intervals(old,new,repetitions=2000,seed=20261108):
    assert [r['document'] for r in old]==[r['document'] for r in new]
    groups=[np.array([i for i,r in enumerate(old) if r['domain']==d]) for d in ['web','code','conversation']]
    values=np.array([[a['student_nll'],b['student_nll'],a['kl'],b['kl'],a['tokens']] for a,b in zip(old,new)])
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(repetitions):
        ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        a,b,c,d,n=values[ix].sum(0);draws.append([math.expm1((b-a)/n),(d-c)/n])
    ci=np.quantile(draws,[.025,.975],axis=0)
    a,b,c,d,n=values.sum(0)
    return {'fresh_vs_old_relative_ppl_change':math.expm1((b-a)/n),
            'fresh_vs_old_relative_ppl_change_ci95':ci[:,0].tolist(),
            'fresh_minus_old_kl':(d-c)/n,'fresh_minus_old_kl_ci95':ci[:,1].tolist(),
            'method':'paired document bootstrap stratified by domain; 2000 draws; lower favors fresh'}


@contextmanager
def mask_mlp(model,permutations,grams,size,keep):
    handles=[]
    try:
        for l,layer in enumerate(model.model.layers):
            def hook(module,args,l=l):
                x=args[0]
                mask=contribution_mask(x,permutations[l],size,keep,grams[l])
                return (x*mask.to(x.dtype),*args[1:])
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
        yield
    finally:
        for h in handles:h.remove()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('artifacts/fresh-groups-quick-v1'))
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('configs/fresh-groups-quick.json').read_text())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=['scripts/quick_fresh_groups.py','pathpred/fresh_groups.py','pathpred/clustering.py',
            'scripts/cluster_neurons.py','pathpred/sparsity_diagnostics.py','pathpred/oracle_checks.py',
            'configs/fresh-groups-quick.json','docs/fresh-groups-protocol.md',
            'data/stage3-corpus-v1/manifest.json','data/stage3-full-v1/manifest.json',
            'data/stage4-clusters-v1/manifest.json','artifacts/sparsity-calibration-v2/grid-results.json']
    fp={s:file_sha256(Path(s)) for s in source}
    mp=a.output/'manifest.json'
    if mp.exists():
        manifest=json.loads(mp.read_text());assert manifest['fingerprint']==fp
        if manifest['status']=='complete':print('Already complete');return
    else:
        manifest={'status':'in_progress','fingerprint':fp,'config':cfg,'layers':[]}
        atomic_json(mp,manifest)
    started=time.monotonic()
    raw,seqs,positions,splits,collection=read_samples(Path('data/stage3-corpus-v1'),Path('data/stage3-full-v1'))
    masks={s:torch.tensor([v==s for v in splits]) for s in ['train','validation','test']}
    manifest['positions_by_split']={k:int(v.sum()) for k,v in masks.items()}
    fresh={size:[] for size in cfg['sizes']};old_orders=[]
    for l in range(raw.shape[0]):
        layer_start=time.monotonic()
        old=load_file(f'data/stage4-clusters-v1/layer-{l:02}.safetensors')['permutation']
        old_orders.append(old)
        x=activation_features(raw[l,masks['train']].cuda())
        entry={'layer':l,'sizes':{}}
        for size in cfg['sizes']:
            target=a.output/f'groups-{size}-layer-{l:02}.safetensors'
            detail=a.output/f'groups-{size}-layer-{l:02}.json'
            if target.exists() and detail.exists():
                d=json.loads(detail.read_text());assert d['fingerprint']==fp
                assert file_sha256(target)==d['file']['sha256']
                perm=load_file(str(target))['permutation']
            else:
                perm,fit=balanced_kmeans(x,size,seed=cfg['seed']+l*100,
                    **{k:cfg[k] for k in ['restarts','max_iterations','relative_tolerance','distance_integer_scale']})
                validate_permutation(perm,raw.shape[-1])
                metrics={}
                for split,mask in masks.items():
                    feat=x if split=='train' else activation_features(raw[l,mask].cuda())
                    metrics[split]={name:grouping_metrics(feat,p,size) for name,p in [('fresh',perm),('old_subdivision',old)]}
                parents=torch.argsort(old)[perm].reshape(-1,size)//128
                cross=(parents!=parents[:,:1]).any(-1)
                d={'fingerprint':fp,'layer':l,'size':size,'fit':fit,'metrics':metrics,
                   'fraction_new_groups_crossing_old_128_boundaries':float(cross.float().mean()),
                   'file':atomic_tensors(target,{'permutation':perm,'stored_indices':original_to_stored_order(perm,old)})}
                atomic_json(detail,d)
            fresh[size].append(perm);entry['sizes'][str(size)]=d
        del x
        if len(manifest['layers'])<=l:manifest['layers'].append(entry)
        atomic_json(mp,manifest)
        print(json.dumps({'layer_complete':l,'seconds':time.monotonic()-layer_start,
                          'elapsed_seconds':time.monotonic()-started}),flush=True)
        if time.monotonic()-started>cfg['maximum_runtime_minutes']*60:
            raise TimeoutError('Quick-run runtime guard; layer artifacts preserved')
    del raw
    fit_seconds=time.monotonic()-started
    # Use exactly the prior windows, including their truncation and ordering.
    previous_manifest=json.loads(Path('artifacts/sparsity-calibration-v2/grid-manifest.json').read_text())
    selected=previous_manifest['selected_documents']
    tokens=load_file('data/stage3-corpus-v1/tokens.safetensors')['input_ids']
    model=load_clustered_qwen('checkpoints/qwen3-1.7b-base-clustered-v1',dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='eager',local_files_only=True)
    permutations={};grams={}
    for size in cfg['sizes']:
        for name in ['old_subdivision','fresh']:
            p=[(torch.arange(6144) if name=='old_subdivision' else original_to_stored_order(fresh[size][l],old_orders[l])).cuda() for l in range(28)]
            permutations[size,name]=p
            grams[size,name]=[contribution_gram(layer.mlp.down_proj.weight,p[l],size) for l,layer in enumerate(model.model.layers)]
    windows=[];checks={'all_on_bitwise_equal':True,'old_window_metrics_match_previous':True,
                      'attention_untouched':True,'weights_not_reordered_or_trained':True}
    evaluation_start=time.monotonic()
    with torch.inference_mode():
        for i,meta in enumerate(selected):
            dest=a.output/f'window-{meta["sequence_id"]}.json'
            if dest.exists():
                doc=json.loads(dest.read_text());assert doc['fingerprint']==fp
                windows.append(doc);continue
            prior=json.loads(Path(f'artifacts/sparsity-calibration-v2/grid-window-{meta["sequence_id"]}.json').read_text())
            length=min(meta['length'],512)
            ids=tokens[meta['sequence_id']:meta['sequence_id']+1,:length].long().cuda()
            teacher=model(ids,use_cache=False).logits
            doc={'fingerprint':fp,'metadata':meta,'conditions':{}}
            if i==0:
                for size in cfg['sizes']:
                    with mask_mlp(model,permutations[size,'fresh'],grams[size,'fresh'],size,1.):
                        all_on=model(ids,use_cache=False).logits
                    assert torch.equal(teacher,all_on)
                    del all_on
            for size in cfg['sizes']:
                for keep in cfg['kept_fractions']:
                    for name in ['old_subdivision','fresh']:
                        with mask_mlp(model,permutations[size,name],grams[size,name],size,keep):
                            student=model(ids,use_cache=False).logits
                        metrics=quality_rows(teacher,student,ids)
                        row={k:float(v.double().sum()) for k,v in metrics.items()}
                        row.update(tokens=length-1,domain=meta['domain'],document=meta['document_sha256'])
                        key=f'g{size}-k{round(keep*100)}-{name}'
                        doc['conditions'][key]=row
                        if name=='old_subdivision':
                            oldrow=prior['conditions'][f'mlp-g{size}-contribution-k{round(keep*100)}-none']
                            for field in ['teacher_nll','student_nll','kl','top1_agreement']:
                                assert row[field]==oldrow[field], (meta['sequence_id'],key,field,row[field],oldrow[field])
                        del student
            atomic_json(dest,doc);windows.append(doc);del teacher
            print(json.dumps({'documents_complete':i+1,'total':len(selected),'evaluation_seconds':time.monotonic()-evaluation_start}),flush=True)
    results={};comparisons={}
    for size in cfg['sizes']:
        for keep in cfg['kept_fractions']:
            pairs={}
            for name in ['old_subdivision','fresh']:
                key=f'g{size}-k{round(keep*100)}-{name}';rows=[d['conditions'][key] for d in windows]
                pairs[name]=rows;overall=aggregate(rows)
                overall.update(bootstrap_quality(rows,cfg['bootstrap_repetitions'],cfg['seed']))
                results[key]={'overall':overall,'domains':{domain:aggregate([r for r in rows if r['domain']==domain]) for domain in ['web','code','conversation']}}
            comparisons[f'g{size}-k{round(keep*100)}']=paired_intervals(pairs['old_subdivision'],pairs['fresh'],cfg['bootstrap_repetitions'],cfg['seed'])
    report={'status':'complete','fingerprint':fp,'config':cfg,'results':results,'paired_comparisons':comparisons,
            'checks':checks,'documents':len(windows),'fit_and_read_seconds':fit_seconds,
            'evaluation_seconds':time.monotonic()-evaluation_start,'total_seconds':time.monotonic()-started}
    atomic_json(a.output/'results.json',report)
    manifest.update(status='complete',total_seconds=report['total_seconds']);atomic_json(mp,manifest)
    print(json.dumps({'status':'complete','seconds':report['total_seconds'],'comparisons':comparisons}),flush=True)


if __name__=='__main__':main()
