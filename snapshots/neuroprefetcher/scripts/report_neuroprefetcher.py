"""Read saved scores only; this module cannot execute the planner."""
import argparse
import json
from pathlib import Path
import numpy as np
from pathpred.collection import atomic_json,file_sha256
from scripts.oracle_quality import aggregate


def paired_intervals(candidate,baseline,repetitions=2000,seed=20261113):
    b={r['document']:r for r in baseline};a={r['document']:r for r in candidate}
    if len(a)!=len(candidate) or len(b)!=len(baseline) or set(a)!=set(b):
        raise ValueError('Paired comparison requires identical unique document IDs')
    ids=sorted(a)
    for key in ids:
        if a[key]['tokens']!=b[key]['tokens'] or a[key]['domain']!=b[key]['domain']:
            raise ValueError('Document scoring windows differ')
        if abs(a[key]['teacher_nll']-b[key]['teacher_nll'])/a[key]['tokens']>1e-4:
            raise ValueError('Dense teacher differs; do not silently pair protocols')
    domains=sorted({r['domain'] for r in candidate});groups=[[k for k in ids if a[k]['domain']==d] for d in domains]
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repetitions):
        selected=[k for group in groups for k in rng.choice(group,len(group),replace=True)]
        x=aggregate([a[k] for k in selected]);y=aggregate([b[k] for k in selected])
        samples.append([x['relative_ppl_increase']-y['relative_ppl_increase'],
                        x['teacher_kl_nats']-y['teacher_kl_nats'],x['top1_agreement']-y['top1_agreement']])
    bounds=np.quantile(samples,[.025,.975],axis=0)
    return {'ppl_relative_increase_difference_ci95':bounds[:,0].tolist(),
        'kl_nats_difference_ci95':bounds[:,1].tolist(),'top1_difference_ci95':bounds[:,2].tolist(),
        'direction':'NeuroPrefetcher minus saved planner','method':'Paired document bootstrap, stratified by domain',
        'repetitions':repetitions,'seed':seed,'documents':len(ids)}


def report(c,planner_root):
    root=Path(c['root']);result={'neuro':{},'planner':{},'paired':{},'missing_results':[],
        'accuracy_checkpoints':{},'not_run_due_to_accuracy_gate':[],
        'planner_read_only':True,'notes':[
            'No planner training or evaluation was run for this comparison.',
            'Budgets are calibrated on validation; actual test retention may differ.',
            '51.62% calibration is a sensitivity check, not forced test-set matching.',
            'Dense masking and modeled overhead do not demonstrate sparse runtime speedup.']}
    for budget in (50,40):
        folder=root/f'budget-{budget}'
        result['accuracy_checkpoints'][str(budget)]=[
            json.loads(path.read_text()) for path in sorted(folder.glob('accuracy-*.json'),
                key=lambda p:int(p.stem.removeprefix('accuracy-')))]
        for field in ('training','accounting','runtime-overhead'):
            path=folder/(field+'.json')
            if path.exists():result['neuro'][f'{budget}-{field}']=json.loads(path.read_text())
        base=planner_root/'autoregressive'/f'budget-{budget}'/'evaluation.json'
        if base.exists():
            result['planner'][str(budget)]={'results':json.loads(base.read_text()),'file':str(base),'sha256':file_sha256(base)}
        else:result['missing_results'].append(str(base))
        for path in sorted(folder.glob('evaluation-*.json')):
            if path.name.endswith('-documents.json'):continue
            value=json.loads(path.read_text());result['neuro'][path.stem]=value
            documents=path.with_name(path.stem+'-documents.json')
            baseline=base.with_name('evaluation-documents.json')
            if documents.exists() and baseline.exists():
                result['paired'][path.stem]=paired_intervals(json.loads(documents.read_text()),json.loads(baseline.read_text()))
    launch=root/'launch/completion.json'
    if launch.exists():result['completion']=json.loads(launch.read_text())
    stopped=any(item.get('status')=='stopped_low_centroid_accuracy'
                for item in result['neuro'].values())
    if stopped:
        result['not_run_due_to_accuracy_gate']=[
            'Remaining budget-specific training and validation calibration',
            '63-document gated quality evaluation (perplexity, KL, top-token agreement and domains)',
            'Active-neuron reuse during gated decoding',
            'Paired document-bootstrap differences versus the saved planner']
    cost=root/'gpu-cost.json'
    if cost.exists():result['gpu_cost']=json.loads(cost.read_text())
    atomic_json(root/'comparison.json',result)
    lines=['# NeuroPrefetcher versus saved upfront planner','',
        '| Model / validation target | Test neurons kept | PPL increase | KL (nats) | Top-token agreement |',
        '|---|---:|---:|---:|---:|']
    for label,item in result['planner'].items():
        o=item['results']['overall'];lines.append(f"| Saved planner / {label}% | {100*o['routed_kept']:.3f}% | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    for label,item in result['neuro'].items():
        if 'overall' not in item:continue
        o=item['overall'];lines.append(f"| NeuroPrefetcher / {label.removeprefix('evaluation-')} | {100*o['routed_kept']:.3f}% | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    lines+=['','Full per-domain results, accuracy diagnostics, cost and paired intervals: `comparison.json`.','']
    for label,item in result['neuro'].items():
        if item.get('status')=='stopped_low_centroid_accuracy':
            a=item['accuracy'];lines.append(f"Stopped at {item['tokens']:,} tokens: gate centroid accuracy {100*a['gate_top1']:.2f}%, up {100*a['up_top1']:.2f}%, below the predeclared90% held-out accuracy gate. No quality result is inferred.")
    for budget,checkpoints in result['accuracy_checkpoints'].items():
        if not checkpoints:continue
        lines+=['',f'## Held-out centroid accuracy: {budget}% vocabulary','',
            '| Training tokens | Gate top-1 | Up top-1 | Most-frequent gate | Most-frequent up |',
            '|---:|---:|---:|---:|---:|']
        for checkpoint in checkpoints:
            majority=[float(np.mean(values)) for values in checkpoint['most_frequent_baseline']]
            lines.append(f"| {checkpoint['tokens_trained']:,} | {100*checkpoint['gate_top1']:.2f}% | {100*checkpoint['up_top1']:.2f}% | {100*majority[0]:.2f}% | {100*majority[1]:.2f}% |")
    if result['not_run_due_to_accuracy_gate']:
        lines+=['','The requested accuracy stop rule prevented the following measurements; they are unavailable, not zero:','',
                *['- '+value for value in result['not_run_due_to_accuracy_gate']]]
    lines+=['']+result['notes'];(root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(root/'comparison.md'),'paired_comparisons':len(result['paired'])}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/neuroprefetcher.json')
    parser.add_argument('--planner-root',type=Path,default=Path('/workspace/pathpred-planner/artifacts/planner-v1'))
    a=parser.parse_args();report(json.loads(Path(a.config).read_text()),a.planner_root)


if __name__=='__main__':main()
