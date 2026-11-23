"""Calibration first; then a frozen component/rule/protection/granularity grid."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from pathpred import gate_qwen3
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.reordering import load_clustered_qwen
from pathpred.oracle_checks import quality_rows
from pathpred.sparsity_diagnostics import Condition, DynamicPolicy, projection
from scripts.oracle_quality import select_documents, aggregate, bootstrap_quality


def grid(cfg):
    result = []
    for component in ['mlp', 'attention', 'both']:
        for size in ([128] if component == 'attention' else cfg['group_sizes']):
            for rule in cfg['rules']:
                for keep in cfg['kept_fractions']:
                    for protection in cfg['protections']:
                        result.append(Condition(component, size, rule, keep, protection))
    return result


@torch.inference_mode()
def audit(model, ids):
    """Independent hook vs wrapper, including cached sparse continuation."""
    ids = ids[:, :32]
    original_dtype=next(model.parameters()).dtype
    policy = DynamicPolicy(model)
    baseline = model(ids, use_cache=False).logits
    cases = [Condition('both', 1, 'magnitude', 1),
             Condition('mlp', 1, 'magnitude', .5),
             Condition('attention', 128, 'magnitude', .5),
             Condition('both', 32, 'contribution', .5)]
    reference = {}
    for c in cases:
        with policy.apply(c,capture=True):
            full = model(ids, use_cache=False).logits
        full_masks={k:v[:,24:].clone() for k,v in policy.latest_masks.items()}
        with policy.apply(c,capture=True):
            prefix = model(ids[:, :24], use_cache=True)
            tail = model(ids[:, 24:], past_key_values=prefix.past_key_values, use_cache=True).logits
        flips=sum(int((full_masks[k]!=v).sum()) for k,v in policy.latest_masks.items())
        count=sum(v.numel() for v in full_masks.values())
        reference[c.name] = (full, tail, flips/count)
    controller = gate_qwen3(model, 1)
    outcomes = {}
    for c in cases:
        with controller.use_gates(), policy.apply(c, controller):
            full = model(ids, use_cache=False).logits
        with controller.use_gates(), policy.apply(c, controller):
            prefix = model(ids[:, :24], use_cache=True)
            tail = model(ids[:, 24:], past_key_values=prefix.past_key_values, use_cache=True).logits
        ref, ref_tail, flips = reference[c.name]
        assert torch.equal(full, ref), 'Wrapper differs from direct masking'
        assert torch.equal(tail, ref_tail), 'Cached wrapper differs from direct masking'
        rows = quality_rows(full[:, 24:], tail, ids[:, 24:])
        outcomes[c.name] = {'wrapper_direct_bitwise_equal': True,
            'cached_wrapper_direct_bitwise_equal': True,
            'cached_vs_full_max_logit_difference': float((full[:,24:]-tail).abs().max()),
            'cached_vs_full_mean_kl': float(rows['kl'].mean()),
            'cached_vs_full_tail_mask_disagreement': flips}
        print(json.dumps({'harness_case': c.name, **outcomes[c.name]}), flush=True)
        if c.keep == 1:
            assert torch.equal(baseline, full), 'All-on is not dense'
    controller.remove()
    if original_dtype==torch.float32:
        assert max(v['cached_vs_full_mean_kl'] for v in outcomes.values()) < 1e-5, 'FP32 cached/full mismatch'
    elif max(v['cached_vs_full_mean_kl'] for v in outcomes.values()) >= .01:
        # Investigate precision-sensitive hard selection on unchanged stock cache
        # logic before allowing a calibration run. Never relax the FP32 check.
        original_buffers={name:value.clone() for name,value in model.named_buffers()}
        model.float()
        fp32=audit(model,ids)
        model.to(original_dtype)
        # Rotary frequencies can be FP32 even in a BF16 model. A blanket .to()
        # would round these buffers; restore their exact original values/dtypes.
        for name,value in original_buffers.items():
            parent,_,leaf=name.rpartition('.')
            model.get_submodule(parent)._buffers[leaf]=value
        restored=model(ids,use_cache=False).logits
        assert torch.equal(restored,baseline), 'Precision audit mutated dense model'
        return {'bf16':outcomes,'fp32':fp32,
                'dense_logits_bitwise_unchanged_after_precision_audit':True,
                'original_buffer_dtypes':{k:str(v.dtype) for k,v in original_buffers.items()},
                'finding':'Direct and wrapped gating agree exactly; BF16 cached/full dynamic selections are precision-sensitive'}
    return outcomes


@torch.inference_mode()
def calibrate_statistics(model, tokens, records, cfg, output):
    selected = select_documents(records, 'train', cfg['calibration_documents_per_domain'], cfg)
    observations = {(l,k): [] for l in range(model.config.num_hidden_layers) for k in ['mlp','attention']}
    handles = []
    for l,k in observations:
        def hook(module, args, key=(l,k)):
            observations[key].append(args[0][0].detach().cpu())
        handles.append(projection(model,l,k).register_forward_pre_hook(hook))
    try:
        for meta in selected:
            n=min(meta['length'],cfg['calibration_sequence_length'])
            ids=tokens[meta['sequence_id']:meta['sequence_id']+1,:n].long().cuda()
            model(ids,use_cache=False)
    finally:
        for h in handles:h.remove()
    tensors={};stats={}
    for (l,k), pieces in observations.items():
        x=torch.cat(pieces).cuda().float()
        cn=projection(model,l,k).weight.float().norm(dim=0)
        for width in (cfg['group_sizes'] if k=='mlp' else [model.config.head_dim]):
            z=x.reshape(len(x),-1,width)
            q=z.square().sum(-1).sqrt().quantile(.75,dim=0)
            mean=(z*cn.reshape(-1,width)).square().sum(-1).mean(0)
            always=mean.topk(math.ceil(len(mean)*cfg['always_on_fraction'])).indices
            key=(l,k,width);stats[key]={'q75':q,'always':always}
            tensors[f'{l}.{k}.{width}.q75']=q.cpu()
            tensors[f'{l}.{k}.{width}.always']=always.cpu()
        observations[l,k]=[]
    entry=atomic_tensors(output/'calibration-statistics.safetensors',tensors)
    atomic_json(output/'calibration-statistics.json',{'documents':selected,'positions':len(x),'file':entry,
        'threshold':'strict above own training q75','always':'top training mean squared WINA score'})
    return stats


def summarize(documents, conditions, cfg):
    results={}
    for c in conditions:
        rows=[d['conditions'][c.name] for d in documents]
        overall=aggregate(rows)
        overall.update(bootstrap_quality(rows,cfg['bootstrap_repetitions'],cfg['seed']))
        results[c.name]={'condition':asdict(c),'overall':overall,
            'domains':{domain:aggregate([r for r in rows if r['domain']==domain])
                       for domain in ['web','code','conversation']},
            'retained':{k:sum(r['retained'][k]*r['tokens'] for r in rows)/sum(r['tokens'] for r in rows)
                        for k in ['mlp','attention']}}
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['calibration','grid'],required=True)
    p.add_argument('--output',type=Path,default=Path('artifacts/sparsity-calibration-v2'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=json.loads(Path('configs/sparsity-calibration.json').read_text())
    source=['configs/sparsity-calibration.json','docs/sparsity-calibration-protocol.md',
            'scripts/calibrate_sparsity.py','pathpred/sparsity_diagnostics.py',
            'pathpred/gated_qwen.py','pathpred/reordering.py','pathpred/oracle_checks.py',
            'data/stage3-corpus-v1/manifest.json']
    fp={s:file_sha256(Path(s)) for s in source}
    if (a.output/f'{a.stage}-results.json').exists():
        prior=json.loads((a.output/f'{a.stage}-results.json').read_text())
        assert prior['fingerprint']==fp
        print('Already complete');return
    if a.stage=='grid':
        prior=json.loads((a.output/'calibration-results.json').read_text())
        assert prior['fingerprint']==fp and prior['status']=='complete'
        # A catastrophic result requires an explicit investigated amendment.
        assert prior['calibration_noncatastrophic'], 'Investigate calibration before grid'
    root=Path('data/stage3-corpus-v1')
    corpus=json.loads((root/'manifest.json').read_text())
    for entry in corpus['files']:assert file_sha256(root/entry['file'])==entry['sha256']
    records=[json.loads(line) for line in (root/'sequences.jsonl').read_text().splitlines()]
    selected=select_documents(records,'test',cfg['test_documents_per_domain'],cfg)
    tokens=load_file(str(root/'tokens.safetensors'))['input_ids']
    model=load_clustered_qwen('checkpoints/qwen3-1.7b-base-clustered-v1',dtype=torch.bfloat16,
                            device_map={'':'cuda'},attn_implementation='eager',local_files_only=True)
    if a.stage=='calibration':
        meta=select_documents(records,'validation',1,cfg)[0]
        ids=tokens[meta['sequence_id']:meta['sequence_id']+1,:32].long().cuda()
        checks=audit(model,ids)
        atomic_json(a.output/'harness-audit.json',checks)
        calibration={}
        conditions=[Condition('mlp',1,'magnitude',b) for b in [.9,.75,.5]]
    else:
        calibration=calibrate_statistics(model,tokens,records,cfg,a.output)
        conditions=grid(cfg)
    controller=gate_qwen3(model,1);policy=DynamicPolicy(model,calibration)
    for c in conditions:policy.prepare(c)
    manifest={'stage':a.stage,'fingerprint':fp,'config':cfg,'selected_documents':selected,
              'conditions':[asdict(c) for c in conditions],'status':'in_progress'}
    atomic_json(a.output/f'{a.stage}-manifest.json',manifest)
    started=time.monotonic();documents=[]
    with torch.inference_mode():
        for index,meta in enumerate(selected):
            path=a.output/f'{a.stage}-window-{meta["sequence_id"]}.json'
            if path.exists():
                doc=json.loads(path.read_text());assert doc['fingerprint']==fp
            else:doc={'metadata':meta,'fingerprint':fp,'conditions':{}}
            length=min(meta['length'],cfg['maximum_sequence_length'])
            ids=tokens[meta['sequence_id']:meta['sequence_id']+1,:length].long().cuda()
            with controller.use_gates():teacher=model(ids,use_cache=False).logits
            for ci,c in enumerate(conditions):
                if c.name in doc['conditions']:continue
                if time.monotonic()-started>cfg['maximum_runtime_hours']*3600:raise TimeoutError('Runtime cap')
                with controller.use_gates(),policy.apply(c,controller):
                    student=model(ids,use_cache=False).logits
                metrics=quality_rows(teacher,student,ids)
                row={k:float(v.double().sum()) for k,v in metrics.items()}
                row.update(tokens=length-1,domain=meta['domain'],document=meta['document_sha256'],
                           retained=policy.fractions(),max_token_kl=float(metrics['kl'].max()))
                doc['conditions'][c.name]=row
                del student
                if (ci+1)%30==0:
                    atomic_json(path,doc)
                    print(json.dumps({'stage':a.stage,'document':index+1,'conditions_done':ci+1,
                                      'conditions_total':len(conditions),'seconds':time.monotonic()-started}),flush=True)
            atomic_json(path,doc);documents.append(doc);del teacher
            print(json.dumps({'stage':a.stage,'completed_documents':index+1,'total':len(selected),
                              'seconds':time.monotonic()-started}),flush=True)
    results=summarize(documents,conditions,cfg)
    report={'status':'complete','stage':a.stage,'fingerprint':fp,'config':cfg,'results':results,
            'documents':len(documents),'seconds':time.monotonic()-started}
    if a.stage=='calibration':
        primary=results[conditions[-1].name]['overall']
        report['calibration_noncatastrophic']=primary['relative_ppl_increase']<1 and primary['teacher_kl_nats']<1
        report['catastrophe_trigger']='>=2x dense PPL or >=1 nat KL at 50% MLP neurons; investigation trigger, not scientific pass criterion'
    atomic_json(a.output/f'{a.stage}-results.json',report)
    manifest['status']='complete';atomic_json(a.output/f'{a.stage}-manifest.json',manifest)
    print(json.dumps({'stage':a.stage,'status':'complete','seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
