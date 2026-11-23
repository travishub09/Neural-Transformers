"""Freeze native thresholds on validation, then evaluate hard gated trajectories."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import load_file
from pathpred.budget_calibration import calibrate_threshold
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.end_to_end import quality_route, dense_teacher
from pathpred.neuron_prediction import NeuronPredictor, neuron_config
from pathpred.oracle_checks import quality_rows
from scripts.neuron_guessing import backbone, data, CAL
from scripts.oracle_quality import select_documents, aggregate, bootstrap_quality

ROOT=Path('artifacts/native-calibration-v1')
SEED=20261111


def selections():
    corpus,records=data('data/stage3-corpus-v1')
    validation=select_documents(records,'validation',24,{'seed':SEED,'minimum_sequence_length':128})
    test=select_documents(records,'test',24,{'seed':20261109,'minimum_sequence_length':128})
    assert len(test)==63
    assert not {r['document_sha256'] for r in validation} & {r['document_sha256'] for r in test}
    return corpus['input_ids'],validation,test


def ids_for(corpus,meta):
    return corpus[meta['sequence_id']:meta['sequence_id']+1,:min(meta['length'],512)].long().cuda()


def retention(trace):
    # Only prediction positions with a scored next token; no padded positions.
    return torch.stack([trace['gates'][l].detach().float().mean(-1) for l in range(2,28)]).mean(0)[0,:-1]


@torch.no_grad()
def calibrate(model,controller,p,q,offsets,corpus,validation,target,output):
    identity={'target':target,'validation_documents':validation,'base_offsets':offsets.cpu().tolist(),
              'policy':'one global score shift; hard gates; closed-loop recomputation; token-weighted mean',
              'selection_seed':SEED,'tolerance':.0005,'maximum_evaluations':16}
    atomic_json(output.with_name(output.stem+'-protocol.json'),identity)
    def measure(shift):
        total=0.;count=0
        for meta in validation:
            ids=ids_for(corpus,meta)
            with quality_route(model,controller,p,q,mode='hard',score_offsets=offsets+shift,
                               checkpoint_layers=False) as trace:
                model.model(ids,use_cache=False)
            r=retention(trace);total+=float(r.double().sum());count+=r.numel()
        used=total/count
        print(json.dumps({'stage':'threshold_search','target':target,'shift':shift,'retention':used}),flush=True)
        return used
    result=calibrate_threshold(measure,target,tolerance=.0005,max_evaluations=16)
    result.update(identity)
    result['offsets']=(offsets+result['shift']).cpu().tolist()
    # Commit before any test quality is measured. Never select by test metrics.
    atomic_json(output,result)
    return offsets+result['shift'],result


def correlation(a,b):
    if np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])


def effort_summary(kept,difficulty,domains):
    out={'mean':float(np.mean(kept)),'std':float(np.std(kept)),
         'quantiles':dict(zip(['min','p05','p25','median','p75','p95','max'],
                             np.quantile(kept,[0,.05,.25,.5,.75,.95,1]).tolist())),
         'pearson_dense_next_token_nll':correlation(kept,difficulty)}
    residual_k=kept.copy();residual_n=difficulty.copy()
    for d in np.unique(domains):
        mask=domains==d;residual_k[mask]-=kept[mask].mean();residual_n[mask]-=difficulty[mask].mean()
    out['pearson_within_domain_centered']=correlation(residual_k,residual_n)
    out['interpretation']='Association with dense next-token surprisal, not causal evidence of useful difficulty adaptation.'
    return out


@torch.no_grad()
def evaluate(model,controller,p,q,offsets,corpus,selected,output):
    rows=[];samples=[];start=time.monotonic()
    for i,meta in enumerate(selected):
        ids=ids_for(corpus,meta);teacher,_=dense_teacher(model,controller,ids)
        with quality_route(model,controller,p,q,mode='hard',score_offsets=offsets,
                           checkpoint_layers=False) as trace:
            student=model(ids,use_cache=False).logits
        for l in range(28):
            gate=trace['gates'][l].detach()
            assert ((gate==0)|(gate==1)).all()
            assert (trace['bits'][l][gate==0]==-1).all()
            if l<2: assert gate.all()
        metrics=quality_rows(teacher,student,ids)
        kept=retention(trace).cpu().double().numpy()
        difficulty=metrics['teacher_nll'].flatten().cpu().double().numpy()
        assert len(kept)==len(difficulty)
        row={k:float(v.double().sum()) for k,v in metrics.items()}
        row.update(tokens=len(kept),document=meta['document_sha256'],domain=meta['domain'],routed_kept=float(kept.mean()))
        rows.append(row)
        samples.append((kept,difficulty,np.full(len(kept),['web','code','conversation'].index(meta['domain'])),np.full(len(kept),i)))
        if (i+1)%10==0: print(json.dumps({'stage':'hard_evaluation','output':output.name,'documents':i+1}),flush=True)
    arrays=[np.concatenate([s[j] for s in samples]) for j in range(4)]
    kept,difficulty,domains,documents=arrays
    def summarize(subrows,mask):
        result=aggregate(subrows);result.update(bootstrap_quality(subrows,2000,SEED))
        result['method']='paired document bootstrap within domains'
        result['routed_kept']=float(kept[mask].mean());result['all_mlp_kept']=(2+26*result['routed_kept'])/28
        result['effort']=effort_summary(kept[mask],difficulty[mask],domains[mask])
        return result
    result={'overall':summarize(rows,np.ones(len(kept),dtype=bool)),
            'domains':{d:summarize([r for r in rows if r['domain']==d],domains==i)
                       for i,d in enumerate(['web','code','conversation'])},
            'seconds':time.monotonic()-start,'documents':len(rows),
            'offsets':offsets.cpu().tolist(),'scope':'Dense masked simulation; attention and first two MLP layers dense; no measured inference speedup.'}
    atomic_json(output,result);atomic_json(output.with_name(output.stem+'-documents.json'),rows)
    atomic_tensors(output.with_suffix('.safetensors'),dict(zip(
        ['retention','dense_next_token_nll','domain','document'],[torch.from_numpy(a) for a in arrays])))
    print(json.dumps({'stage':'evaluation_complete','output':output.name,'overall':result['overall']}),flush=True)
    return result


def main():
    global ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--signal',choices=['binary','hidden'],default='binary')
    parser.add_argument('--pilot-root',type=Path,default=Path('artifacts/e2e-pilot-v1'))
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args();ROOT=args.output
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    corpus,validation,test=selections()
    atomic_json(ROOT/'selection.json',{'validation':validation,'test':test,'disjoint':True})
    model,controller=backbone();q=load_file(str(CAL),device='cuda')['q75']
    p=NeuronPredictor(neuron_config(args.signal)).cuda().eval()
    for label,target in [('2000000',.5),('3000000',.4)]:
        meta=json.loads((args.pilot_root/f'checkpoint-{label}.json').read_text())
        checkpoint=args.pilot_root/f'predictor-{label}.safetensors'
        assert file_sha256(checkpoint)==meta['file']['sha256']
        p.load_state_dict(load_file(str(checkpoint),device='cuda'))
        offsets=torch.tensor(meta['offsets'],device='cuda')
        path=ROOT/f'threshold-{int(target*100)}.json'
        if path.exists():
            saved=json.loads(path.read_text());assert saved['checkpoint_sha256']==file_sha256(checkpoint)
            offsets=torch.tensor(saved['offsets'],device='cuda')
        else:
            offsets,result=calibrate(model,controller,p,q,offsets,corpus,validation,target,path)
            result['checkpoint_sha256']=file_sha256(checkpoint);atomic_json(path,result)
        output=ROOT/f'evaluation-{int(target*100)}.json'
        if not output.exists(): evaluate(model,controller,p,q,offsets,corpus,test,output)
    atomic_json(ROOT/'completion.json',{'status':'complete'})


if __name__=='__main__': main()
