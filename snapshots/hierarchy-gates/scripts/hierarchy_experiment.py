"""Matched A/B/C/D hidden-state hierarchy pilot; no GPU lifecycle operations."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json,atomic_tensors,file_sha256
from pathpred.neuron_prediction import NeuronPredictor,neuron_config
from pathpred.end_to_end import quality_route,dense_teacher,quality_loss,DualBudget
from pathpred.hierarchical import HierarchicalPredictor,hierarchy_route,DecodeCost,gate_accounting
from pathpred.budget_calibration import calibrate_threshold
from pathpred.oracle_checks import quality_rows
from scripts.neuron_guessing import backbone,data,CAL
from scripts.calibrate_native import selections,ids_for
from scripts.oracle_quality import aggregate,bootstrap_quality
from scripts.combined_experiment import choose_windows


def protocol(path):
    c=json.loads(Path(path).read_text())
    assert c['gate_recipe']=='hard-threshold-ste' and c['layer_gate_scope']=='mlp-only'
    assert c['sequence_length']==512 and c['group_size']==128
    assert c['milestones']==sorted(set(c['milestones'])) and c['milestones'][-1]==c['train_tokens']
    return c


def warm(budget,variant,device='cuda'):
    root=Path(f'artifacts/e2e-hidden-full-v1/budget-{budget}')
    checkpoint=root/'predictor-10000000.safetensors'
    meta=json.loads((root/'checkpoint-10000000.json').read_text())
    assert file_sha256(checkpoint)==meta['file']['sha256']
    p=HierarchicalPredictor(neuron_config('hidden'),variant).to(device).eval()
    p.load_hidden_state(load_file(str(checkpoint),device=device))
    offsets=torch.tensor(json.loads((root/'threshold-10000000.json').read_text())['offsets'],device=device)
    return p,offsets,meta


def training_data(c):
    corpus,records=data(c['training_corpus']);forbidden=set()
    for path in c['exclude_corpora']:
        _,old=data(path);forbidden.update(x['document_sha256'] for x in old)
    order=choose_windows(records,forbidden,c['milestones'],c['data_seed'])
    audit={'window_indices':order,'tokens_per_branch':c['train_tokens'],'branches':8,
           'same_windows_and_order':True,'excluded_documents':len(forbidden),
           'unique_documents':len({records[i]['document_sha256'] for i in order}),
           'scope':'Fresh relative to original warm starts; may overlap paused combined continuation data, which none of these warm starts used.'}
    path=Path(c['root'])/'data-audit.json'
    if path.exists():assert json.loads(path.read_text())==audit
    else:atomic_json(path,audit)
    return corpus['input_ids'],records,order


def fingerprint(config_path,budget):
    c=protocol(config_path)
    names=[config_path,'pathpred/hierarchical.py','pathpred/end_to_end.py','pathpred/neuron_prediction.py',
           'pathpred/path_model.py','pathpred/gated_qwen.py','pathpred/reordering.py',
           'scripts/hierarchy_experiment.py','scripts/calibrate_native.py','scripts/neuron_guessing.py',
           'scripts/oracle_quality.py','scripts/combined_experiment.py','pathpred/budget_calibration.py',
           str(Path(c['root'])/'data-audit.json'),str(Path(c['training_corpus'])/'manifest.json'),
           'data/stage3-corpus-v1/manifest.json',str(CAL),
           f'artifacts/e2e-hidden-full-v1/budget-{budget}/checkpoint-10000000.json',
           f'artifacts/e2e-hidden-full-v1/budget-{budget}/threshold-10000000.json']
    return {str(n):file_sha256(Path(n)) for n in names}


def train_batch(model,controller,p,q,offsets,ids,optimizer,dual,cost,c,seen):
    optimizer.zero_grad(set_to_none=True)
    factor=min(1.,(seen+ids.numel())/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/c['train_tokens'])))
    for g in optimizer.param_groups:g['lr']=c['learning_rate']*factor*g['multiplier']
    temperature=.3*(.1/.3)**min(1.,seen/c['train_tokens'])
    target,teacher_pre=dense_teacher(model,controller,ids)
    with hierarchy_route(model,controller,p,q,score_offsets=offsets,temperature=temperature) as trace:
        h=model.model(ids,use_cache=False).last_hidden_state
        quality,metrics=quality_loss(model,h,trace['pre_norm'],target,teacher_pre,trace,
            alpha=0.,beta=c['beta'],epsilon=c['epsilon'],chunk_size=c['teacher_head_chunk_size'])
        levels=gate_accounting(trace);used=levels['used'].mean();total=cost.fraction(used)
        loss=quality+dual.alpha*total
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite hierarchy loss')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True)
        assert all(not v.requires_grad and v.grad is None for v in model.parameters())
        values={k:float(v.detach()) for k,v in metrics.items()}
        values.update({k:float(v.detach().mean()) for k,v in levels.items()})
        values.update(loss=float(loss.detach()),total_compute=float(total.detach()),alpha=dual.alpha,
                      target_total_compute=dual.target,gradient_norm=float(gn),temperature=temperature)
        optimizer.step()
    dual.update(values['total_compute'])
    return values


@torch.no_grad()
def calibrate(model,controller,p,q,offsets,cost,target,corpus,validation,path):
    target_retention=cost.retained_for(target)
    identity={'target_total_compute':target,'target_effective_retention':target_retention,
              'base_offsets':offsets.cpu().tolist(),'validation_documents':validation,
              'cost':asdict(cost),'policy':'One shared threshold shift applied to neuron, group and layer logits; full closed-loop trajectory recomputed.'}
    atomic_json(path.with_name(path.stem+'-protocol.json'),identity)
    def measure(shift):
        total=0.;count=0
        for meta in validation:
            ids=ids_for(corpus,meta)
            with hierarchy_route(model,controller,p,q,score_offsets=offsets,global_shift=shift,checkpoint_layers=False) as trace:
                model.model(ids,use_cache=False)
            r=gate_accounting(trace,scored_only=True)['used'].mean(-1)
            total+=float(r.double().sum());count+=r.numel()
        used=total/count
        print(json.dumps({'stage':'calibration','shift':shift,'retention':used,'total_compute':cost.fraction(used)}),flush=True)
        return used
    result=calibrate_threshold(measure,target_retention,tolerance=.0005,max_evaluations=20)
    result.update(identity);result['validation_total_compute']=cost.fraction(result['validation_retention'])
    atomic_json(path,result)
    return result


@torch.no_grad()
def evaluate(model,controller,p,q,offsets,shift,cost,corpus,test,path):
    rows=[];started=time.monotonic()
    for i,meta in enumerate(test):
        ids=ids_for(corpus,meta);teacher,_=dense_teacher(model,controller,ids)
        with hierarchy_route(model,controller,p,q,score_offsets=offsets,global_shift=shift,checkpoint_layers=False) as trace:
            student=model(ids,use_cache=False).logits
        for g in trace['gates'].values():assert ((g==0)|(g==1)).all()
        metrics=quality_rows(teacher,student,ids);a=gate_accounting(trace,scored_only=True)
        row={k:float(v.double().sum()) for k,v in metrics.items()}
        row.update(tokens=ids.shape[1]-1,document=meta['document_sha256'],domain=meta['domain'],
                   routed_kept=float(a['used'].mean()),total_compute=float(cost.fraction(a['used'].mean())),
                   levels={k:float(v.mean()) for k,v in a.items()},
                   per_layer={k:v.mean((0,1)).cpu().tolist() for k,v in a.items()})
        rows.append(row)
        if (i+1)%10==0:print(json.dumps({'stage':'evaluation','documents':i+1,'output':str(path)}),flush=True)
    def summary(rs):
        o=aggregate(rs);o.update(bootstrap_quality(rs,2000,20261113));n=o['tokens']
        for k in ('routed_kept','total_compute'):o[k]=sum(r[k]*r['tokens'] for r in rs)/n
        o['levels']={k:sum(r['levels'][k]*r['tokens'] for r in rs)/n for k in rows[0]['levels']}
        o['per_layer']={k:(sum(np.array(r['per_layer'][k])*r['tokens'] for r in rs)/n).tolist() for k in rows[0]['per_layer']}
        return o
    result={'overall':summary(rows),'domains':{d:summary([r for r in rows if r['domain']==d]) for d in ('web','code','conversation')},
            'seconds':time.monotonic()-started,'cost':asdict(cost),'global_shift':shift,
            'base_offsets':offsets.cpu().tolist(),'scope':'Dense masked simulation; modeled batch-1 decode FLOPs at reference context; no measured sparse speedup.'}
    atomic_json(path,result);atomic_json(path.with_name(path.stem+'-documents.json'),rows)
    print(json.dumps({'stage':'evaluation_complete','output':str(path),'overall':{k:result['overall'][k] for k in ('relative_ppl_increase','routed_kept','total_compute')}}),flush=True)
    return result


def checkpoint_eval(model,controller,p,q,offsets,dual,cost,c,root,seen,fp,corpus,validation,test):
    label=f'{seen:08d}';meta=root/f'checkpoint-{label}.json'
    if not meta.exists():
        entry=atomic_tensors(root/f'predictor-{label}.safetensors',{k:v.detach().cpu().contiguous() for k,v in p.state_dict().items()})
        atomic_json(meta,{'tokens':seen,'file':entry,'fingerprint':fp,'dual':asdict(dual),'cost':asdict(cost),
                          'variant':p.variant,'base_offsets':offsets.cpu().tolist()})
    threshold=root/f'threshold-{label}.json'
    if threshold.exists():t=json.loads(threshold.read_text())
    else:t=calibrate(model,controller,p,q,offsets,cost,dual.target,corpus,validation,threshold)
    output=root/f'evaluation-{label}.json'
    if not output.exists():evaluate(model,controller,p,q,offsets,t['shift'],cost,corpus,test,output)


def optimizer_for(p,c):
    coarse={'layer_weight','layer_bias','group_weight','group_bias'}
    groups=[{'params':[v for n,v in p.named_parameters() if n not in coarse],'multiplier':1.}]
    values=[v for n,v in p.named_parameters() if n in coarse]
    if values:groups.append({'params':values,'multiplier':c['coarse_lr_multiplier']})
    return torch.optim.AdamW(groups,lr=c['learning_rate'],weight_decay=c['weight_decay'])


def run_branch(model,controller,q,c,config_path,variant,budget,corpus,records,order,eval_corpus,validation,test):
    root=Path(c['root'])/variant/f'budget-{budget}';root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(c['seed']);p,offsets,meta=warm(budget,variant)
    cost=DecodeCost.from_model(model,p,c['reference_context'])
    baseline=HierarchicalPredictor(neuron_config('hidden'),'A');basecost=DecodeCost.from_model(model,baseline,c['reference_context']);del baseline
    target=basecost.fraction(budget/100);dual=DualBudget(target,meta['dual']['alpha']*cost.dense/cost.variable,c['dual_learning_rate'],c['maximum_alpha'])
    optimizer=optimizer_for(p,c);fp=fingerprint(config_path,budget);seen=position=last_save=0;elapsed=0.
    if (root/'last.pt').exists():
        s=torch.load(root/'last.pt',map_location='cuda',weights_only=True);assert s['fingerprint']==fp
        p.load_state_dict(s['predictor']);optimizer.load_state_dict(s['optimizer']);dual=DualBudget(**s['dual'])
        seen=last_save=s['tokens'];position=s['position'];elapsed=s['elapsed'];offsets=s['offsets']
        torch.set_rng_state(s['cpu_rng'].cpu());torch.cuda.set_rng_state(s['cuda_rng'].cpu());del s
    atomic_json(root/'accounting.json',{'predictor':p.accounting(),'cost':asdict(cost),'baseline_cost':asdict(basecost),
        'target_total_compute':target,'equivalent_retention':cost.retained_for(target),'warm_checkpoint':meta['file']})
    if seen in c['milestones']:checkpoint_eval(model,controller,p,q,offsets,dual,cost,c,root,seen,fp,eval_corpus,validation,test)
    stop=[]
    def request_stop(signum,frame):stop.append(signum)
    old=signal.signal(signal.SIGTERM,request_stop);start=time.monotonic();compute_seconds=0.;new_tokens=0
    def save():
        tmp=root/'last.tmp'
        torch.save({'predictor':p.state_dict(),'optimizer':optimizer.state_dict(),'dual':asdict(dual),'offsets':offsets,
            'tokens':seen,'position':position,'elapsed':elapsed+time.monotonic()-start,'fingerprint':fp,
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},tmp);tmp.replace(root/'last.pt')
    try:
        for index in range(position,len(order)):
            if seen>=c['train_tokens']:break
            length=min(512,min(x for x in c['milestones'] if x>seen)-seen)
            row=records[order[index]];ids=corpus[row['sequence_id']:row['sequence_id']+1,:length].long().cuda()
            torch.cuda.synchronize();tick=time.monotonic()
            values=train_batch(model,controller,p,q,offsets,ids,optimizer,dual,cost,c,seen)
            torch.cuda.synchronize();compute_seconds+=time.monotonic()-tick;seen+=length;new_tokens+=length;position=index+1
            if index%20==0 or seen in c['milestones']:
                progress={'variant':variant,'budget_label':budget,'tokens':seen,'target_total_compute':target,
                          'tokens_per_second':new_tokens/compute_seconds,'elapsed_seconds':elapsed+time.monotonic()-start,'metrics':values}
                atomic_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
            timeout=elapsed+time.monotonic()-start>c['maximum_branch_hours']*3600
            if seen-last_save>=c['checkpoint_every_tokens'] or seen in c['milestones'] or stop or timeout:
                save();last_save=seen
            if stop:
                atomic_json(root/'paused.json',{'tokens':seen,'signal':stop[-1]});return 'paused'
            if timeout:raise TimeoutError('Hierarchy branch time limit reached; resumable state saved')
            if seen in c['milestones']:checkpoint_eval(model,controller,p,q,offsets,dual,cost,c,root,seen,fp,eval_corpus,validation,test)
        assert seen==c['train_tokens']
        atomic_json(root/'completion.json',{'status':'complete','tokens':seen,'fingerprint':fp})
        return 'complete'
    finally:signal.signal(signal.SIGTERM,old)


def smoke(model,controller,q,c,corpus,records,order):
    ids=corpus[records[order[0]]['sequence_id']:records[order[0]]['sequence_id']+1,:512].long().cuda()
    results={}
    for budget in (50,40):
        base,offsets,_=warm(budget,'A')
        original=NeuronPredictor(neuron_config('hidden')).cuda().eval();original.load_state_dict(base.state_dict())
        with torch.no_grad(),quality_route(model,controller,original,q,mode='hard',score_offsets=offsets,checkpoint_layers=False) as bt:
            expected=model(ids,use_cache=False).logits
        del original
        basecost=DecodeCost.from_model(model,base,c['reference_context'])
        for variant in 'ABCD':
            p,off,meta=warm(budget,variant)
            with torch.no_grad(),hierarchy_route(model,controller,p,q,score_offsets=off,checkpoint_layers=False) as tr:
                actual=model(ids,use_cache=False).logits
            assert torch.equal(actual,expected),f'{variant} failed exact-open BF16 logits at {budget}'
            assert all(torch.equal(tr['gates'][l],bt['gates'][l]) for l in range(28))
            cost=DecodeCost.from_model(model,p,c['reference_context']);dual=DualBudget(basecost.fraction(budget/100))
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
            optimizer=optimizer_for(p,c)
            metrics=train_batch(model,controller,p,q,off,ids,optimizer,dual,cost,c,0)
            torch.cuda.synchronize();duration=time.monotonic()-tick
            results[f'{variant}-{budget}']={'warm_logits_exact':True,'warm_masks_exact':True,'step_seconds':duration,
                'tokens_per_second':512/duration,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'metrics':metrics,'cost':asdict(cost),'predictor':p.accounting()}
            print(json.dumps({'smoke':f'{variant}-{budget}','tokens_per_second':512/duration}),flush=True)
            del p,optimizer,actual,tr;torch.cuda.empty_cache()
        del base,expected,bt;torch.cuda.empty_cache()
    atomic_json(Path(c['root'])/'gpu-smoke.json',results)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--variant',choices=list('ABCD'));parser.add_argument('--budget',type=int,choices=[50,40]);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();c=protocol(a.config);Path(c['root']).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    corpus,records,order=training_data(c);model,controller=backbone();q=load_file(str(CAL),device='cuda')['q75']
    if a.smoke:smoke(model,controller,q,c,corpus,records,order);return
    if a.variant is None or a.budget is None:parser.error('Training needs --variant and --budget')
    eval_corpus,validation,test=selections()
    result=run_branch(model,controller,q,c,a.config,a.variant,a.budget,corpus,records,order,eval_corpus,validation,test)
    if result=='paused':raise SystemExit(75)


if __name__=='__main__':main()
