"""Matched-budget quality comparisons using saved checkpoints only."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json,file_sha256
from pathpred.budget_calibration import calibrate_threshold
from scripts import mistral_planner as mp,mistral_full as mf,planner_experiment as pe
from scripts.neuron_guessing import data
from scripts.oracle_quality import aggregate,bootstrap_quality
from scripts.mistral_comparison_core import Policy,ReleasedNeuro,score_batch
ROOT=Path('artifacts/mistral-comparisons-v1')

def config():return json.loads(Path('configs/mistral-planner-phased.json').read_text())

def heartbeat(i,n):print(json.dumps(dict(stage='comparison_decode',position=i,total=n)),flush=True)

def load(kind,budget,prefix=2):
    if budget in (30,20):
        assert kind=='planner','Lower budgets currently support saved planner checkpoints only'
        c=json.loads(Path('configs/mistral-lower-budgets.json').read_text())
    else:c=config()
    model,ctrl=mp.load_model(c);p=q=offsets=neuro=None
    if kind=='planner':
        family='mistral-lower-v1' if budget in (30,20) else 'mistral-phased-v1'
        root=Path(f'artifacts/{family}/budget-{budget}')
        done=json.loads((root/'completion.json').read_text());p=mp.make_planner(c)
        pe.load_stage_weights(p,root,done['tokens']);p.eval()
        q=load_file(str(mf.PILOT/'thresholds.safetensors'),device='cuda')['q75']
        state=json.loads((root/f"check-{done['tokens']:08d}"/'validation.json').read_text())
        offsets=torch.tensor(state['offsets'],device='cuda')
    if kind=='neuro':neuro=ReleasedNeuro('/workspace/neuroprefetcher-reference')
    return Policy(model,ctrl,kind,prefix=prefix,keep=budget/100,planner=p,q=q,offsets=offsets,neuro=neuro)

def rows(split):
    tensors,metadata=data(config()['heldout_corpus']);chosen=[r for r in metadata if r['split']==split]
    return [(r,tensors['input_ids'][r['sequence_id'],:r['length']].tolist()) for r in chosen]

def score_rows(policy,items,prompt=0,quality=True,batch_size=8):
    documents=[];ks=ex=0.
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size];seq=[s for r,s in batch]
        z=score_batch(policy,seq,[prompt]*len(seq),teacher=quality,progress=heartbeat)
        ks+=float(z['kept_sum'].sum());ex+=float(z['executions'].sum())
        for i,(r,s) in enumerate(batch):
            v=dict(zip(('teacher_nll','student_nll','kl','top1_agreement','tokens'),z['sums'][i].tolist()))
            v.update(document=r['document_sha256'],domain=r['domain'],
                kept_sum=float(z['kept_sum'][i]),sparse_executions=float(z['executions'][i]),
                reused_neurons=float(z['reused'][i]),reuse_denominator=float(z['selected_for_reuse'][i]))
            v['routed_kept']=v['kept_sum']/max(1,v['sparse_executions']);documents.append(v)
    return ks/max(1,ex),documents

def calibration_dir(prefix,prompt,budget):return ROOT/f'prefix-{prefix}-prompt-{prompt}'/f'budget-{budget}'

def calibrate(kind,budget,prefix,prompt):
    out=calibration_dir(prefix,prompt,budget);out.mkdir(parents=True,exist_ok=True)
    path=out/f'{kind}-calibration.json'
    if path.exists():return
    pol=load(kind,budget,prefix);items=rows('validation');trials_path=out/f'{kind}-trials.json'
    trials=json.loads(trials_path.read_text()) if trials_path.exists() else {}
    def measure(value):
        key=str(value)
        if key in trials:return trials[key]
        if kind=='neuro':pol.neuro.cap=int(value)
        else:pol.offsets=base+value
        used,_=score_rows(pol,items,prompt,quality=False)
        trials[key]=used;atomic_json(trials_path,trials)
        print(json.dumps(dict(stage='calibration',kind=kind,value=value,retention=used)),flush=True)
        return used
    if kind=='neuro':
        # Native s30 release cannot reach exactly 50% because some intersections are smaller.
        # Use its native validation retention as the common 50%-nominal target.
        if budget==50:cap=7168;retention=measure(cap);target=retention
        else:
            target=budget/100;lo=1;hi=7168
            while hi-lo>1:
                mid=(lo+hi)//2;used=measure(mid)
                if abs(used-target)<=.0005:lo=hi=mid;break
                if used<target:lo=mid
                else:hi=mid
            candidates=[(x,measure(x)) for x in sorted(set((lo,hi)))]
            cap,retention=min(candidates,key=lambda x:abs(x[1]-target))
            assert abs(retention-target)<=.0005
        record=dict(kind=kind,cap=cap,target=target,validation_retention=retention,
            native_release=(cap==7168),adaptation='none' if cap==7168 else 'validation-calibrated active-index cap; unchanged predictor and centroid bank')
    else:
        reference=json.loads((out/'neuro-calibration.json').read_text());target=reference['validation_retention'];base=pol.offsets.clone()
        result=calibrate_threshold(measure,target,tolerance=.0005,max_evaluations=20)
        record=dict(kind=kind,target=target,validation_retention=result['validation_retention'],
            offsets=(base+result['shift']).cpu().tolist(),calibration=result)
    atomic_json(path,dict(record,split='validation',documents=len(items),prefix=prefix,prompt_tokens=prompt))

def apply_calibration(pol,budget,prompt):
    root=calibration_dir(pol.prefix,prompt,budget)
    if pol.kind=='dense':return
    if pol.kind=='griffin':
        ref=json.loads((root/'neuro-calibration.json').read_text());pol.keep=ref['validation_retention']
    else:
        ref=json.loads((root/f'{pol.kind}-calibration.json').read_text())
        if pol.kind=='neuro':pol.neuro.cap=ref['cap']
        else:pol.offsets=torch.tensor(ref['offsets'],device='cuda')
        pol.keep=ref['target']

def evaluate(kind,budget,prefix,prompt,wikitext=False):
    root=calibration_dir(prefix,prompt,budget)/('wikitext' if wikitext else 'documents');root.mkdir(parents=True,exist_ok=True)
    dest=root/f'{kind}.json'
    if dest.exists():return
    pol=load(kind,budget,prefix);apply_calibration(pol,budget,prompt)
    if wikitext:
        seq=json.loads((ROOT/'wikitext-reference.json').read_text())['sequence']
        items=[(dict(document_sha256='released-wikitext-first-2048',domain='wikitext'),seq)]
    else:items=rows('test')
    tick=time.time();kept,docs=score_rows(pol,items,prompt)
    overall=aggregate(docs);overall.update(routed_kept=kept,
        previous_token_active_reuse=sum(x['reused_neurons'] for x in docs)/max(1,sum(x['reuse_denominator'] for x in docs)))
    if not wikitext:overall.update(bootstrap_quality(docs,2000,20261119))
    atomic_json(root/f'{kind}-documents.json',docs)
    network=pol.planner if kind=='planner' else pol.neuro.net if kind=='neuro' else None
    predictor_params=0 if network is None else sum(x.numel() for x in network.parameters())
    backbone_params=sum(x.numel() for x in pol.model.parameters())
    predictor_bytes=0 if network is None else sum(x.numel()*x.element_size() for x in network.parameters())
    atomic_json(dest,dict(overall=overall,domains={d:aggregate([x for x in docs if x['domain']==d]) for d in sorted({x['domain'] for x in docs})},
        method=kind,budget_nominal=budget,prefix=prefix,prompt_tokens=prompt,seconds=time.time()-tick,
        accounting='all three MLP projections eligible; dense masked quality harness; attention always dense',
        backbone_parameters=backbone_params,predictor_parameters=predictor_params,
        predictor_parameter_share=predictor_params/backbone_params,predictor_stored_weight_bytes=predictor_bytes,
        predictor_weight_bytes_share=predictor_bytes/sum(x.numel()*x.element_size() for x in pol.model.parameters()),
        elapsed_is_quality_evaluation_not_decode_speed_benchmark=True))

def paired(first,second,seed=20261119):
    a={r['document']:r for r in first};b={r['document']:r for r in second};assert a.keys()==b.keys()
    keys=sorted(a)
    for key in keys:
        assert a[key]['tokens']==b[key]['tokens']
        assert abs(a[key]['teacher_nll']-b[key]['teacher_nll'])<.01,'Dense denominators differ'
    def diff(indices):
        aa=aggregate([a[keys[i]] for i in indices]);bb=aggregate([b[keys[i]] for i in indices])
        return [(aa['relative_ppl_increase']-bb['relative_ppl_increase'])*100,aa['teacher_kl_nats']-bb['teacher_kl_nats']]
    rng=np.random.default_rng(seed);samples=np.array([diff(rng.integers(len(keys),size=len(keys))) for _ in range(2000)])
    return dict(order='planner minus baseline; negative favors planner',ppl_percentage_point_gap=diff(range(len(keys)))[0],
        ppl_percentage_point_gap_ci95=np.quantile(samples[:,0],[.025,.975]).tolist(),kl_gap_ci95=np.quantile(samples[:,1],[.025,.975]).tolist(),documents=len(keys))

def report():
    summary={}
    for p in ROOT.glob('prefix-*/budget-*/documents/planner-documents.json'):
        for kind in ('neuro','griffin'):
            b=p.with_name(kind+'-documents.json')
            if b.exists():summary[str(p.parent/ kind)]=paired(json.loads(p.read_text()),json.loads(b.read_text()))
    atomic_json(ROOT/'paired-comparisons.json',summary)
    hella={};dense_path=ROOT/'hellaswag/dense-examples.jsonl'
    if dense_path.exists():
        baseline={x['id']:x for x in map(json.loads,dense_path.read_text().splitlines())}
        rng=np.random.default_rng(20261119)
        for path in (ROOT/'hellaswag').glob('*-examples.jsonl'):
            method={x['id']:x for x in map(json.loads,path.read_text().splitlines())}
            if method.keys()!=baseline.keys() or len(method)!=10042:continue
            delta=np.array([method[i]['accuracy_normalized']-baseline[i]['accuracy_normalized'] for i in sorted(method)])
            values,counts=np.unique(delta,return_counts=True)
            draws=rng.multinomial(len(delta),counts/len(delta),size=2000)@values/len(delta)
            hella[path.stem]=dict(accuracy_normalized_gap_vs_dense=float(delta.mean()),
                paired_ci95=np.quantile(draws,[.025,.975]).tolist(),examples=len(delta))
    atomic_json(ROOT/'hellaswag-paired.json',hella)
    lines=['# Mistral matched comparisons','',
        'Saved final checkpoints; validation-only matching. Dense masked quality, not sparse speed.','',
        '| Protocol | Method | PPL change | Actual suffix kept | KL | Top-token agreement |',
        '|---|---|---:|---:|---:|---:|']
    for path in sorted(ROOT.glob('prefix-*/budget-*/*/*.json')):
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or 'overall' not in value:continue
        v=value['overall'];name=str(path.relative_to(ROOT))
        lines.append(f"| {name} | {value['method']} | {100*v['relative_ppl_increase']:.3f}% | {100*v['routed_kept']:.3f}% | {v['teacher_kl_nats']:.5f} | {100*v['top1_agreement']:.2f}% |")
    lines+=['','## HellaSwag','','| Method | Examples | Accuracy | Normalized accuracy | Actual suffix kept |','|---|---:|---:|---:|---:|']
    for path in sorted((ROOT/'hellaswag').glob('*.json')):
        v=json.loads(path.read_text())
        if 'accuracy_normalized' not in v:continue
        lines.append(f"| {v['method']} | {v['examples']} | {100*v['accuracy']:.2f}% | {100*v['accuracy_normalized']:.2f}% | {100*v['suffix_kept']:.3f}% |")
    lines+=['','Paired intervals and per-document/example outputs are retained beside this report.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibrate','evaluate','wikitext','report'])
    ap.add_argument('--kind',choices=['planner','neuro','griffin'],default='planner');ap.add_argument('--budget',type=int,choices=[50,40],default=50)
    ap.add_argument('--prefix',type=int,choices=[2,20],default=2);ap.add_argument('--prompt',type=int,choices=[0,64],default=0)
    a=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.stage=='report':report()
    elif a.stage=='calibrate':calibrate(a.kind,a.budget,a.prefix,a.prompt)
    else:evaluate(a.kind,a.budget,a.prefix,a.prompt,a.stage=='wikitext')
if __name__=='__main__':main()
