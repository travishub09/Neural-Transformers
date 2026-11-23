"""Matched continuation training and exact causal evaluation for a sequence prior."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, file_sha256
from pathpred.open_loop import PlannerConfig
from pathpred.sequence_prior import SequencePriorPlanner, SequenceTracker, TrackerConfig
from pathpred.sequence_training import end_to_end_batch, prompt_teacher
from pathpred.sequence_decode import decode_batch
from pathpred.end_to_end import DualBudget
from pathpred.budget_calibration import calibrate_threshold
from scripts import planner_experiment as pe
from scripts.neuron_guessing import backbone, data, CAL
from scripts.calibrate_native import selections
from scripts.oracle_quality import select_documents, aggregate, bootstrap_quality

ROOT = Path('artifacts/sequence-prior-v1')


def tracker_config(c):
    return TrackerConfig(prompt_weight=c['prompt_weight'])


def progress(token, total):
    if pe.PAUSE or (ROOT/'PAUSE').exists(): raise SystemExit(75)
    print(json.dumps({'decode_position': token, 'positions': total}), flush=True)


def load_planner(c, variant, budget):
    source = Path(c['source_root'])/f'budget-{budget}'
    meta = json.loads((source/'checkpoint-02000000.json').read_text())
    path = source/meta['file']['file']
    assert file_sha256(path) == meta['file']['sha256']
    p = SequencePriorPlanner(PlannerConfig(width=c['width'], depth=c['depth'], heads=c['heads']),
                             enabled=variant not in ('control', 'baseline')).cuda()
    p.initialize_from(load_file(str(path), device='cuda'), prior_first=variant=='prior_first')
    offsets = torch.tensor(json.loads((source/'threshold.json').read_text())['offsets'], device='cuda')
    return p, offsets, {'path': str(path), 'sha256': meta['file']['sha256']}


def fingerprint(config, variant, budget, source):
    files = list(Path('pathpred').glob('*.py')) + [config, Path(__file__),
             Path('scripts/planner_experiment.py'), Path('scripts/neuron_guessing.py'),
             Path('scripts/calibrate_native.py'), Path('scripts/oracle_quality.py'), CAL,
             Path('data/combined-corpus-v1/manifest.json'), Path('data/stage3-corpus-v1/manifest.json')]
    return {'variant': variant, 'budget': budget, 'source': source,
            'files': {str(f): file_sha256(f) for f in sorted(set(files))}}


def training_data(c):
    pe.ROOT = ROOT
    corpus, records, calibration, order = pe.ordered_data(c, quality=True)
    # Original training split at 1M: two partial windows were consumed, so
    # ceil(total/512) would accidentally reuse its final training window.
    positions = [json.loads((Path(c['source_root'])/f'budget-{b}'/'progress.json').read_text())['position']
                 for b in c['budgets']]
    skipped = max(positions)
    assert skipped >= math.ceil(c['previous_quality_tokens']/c['sequence_length'])
    order = order[skipped:]
    assert len(order)*c['sequence_length'] >= c['e2e_tokens_per_budget']
    atomic_json(ROOT/'continuation-data-audit.json', {
        'previous_training_windows_skipped': skipped, 'available_fresh_windows': len(order),
        'order_sha256': hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        'same_order_all_variants_and_budgets': True, 'heldout_excluded': True})
    return corpus, records, calibration, order


@torch.no_grad()
def prior_offsets(model, controller, p, c, corpus, records, windows, keep):
    values = []
    for index in windows:
        ids = pe.ids_for(corpus, records, index, c['sequence_length'])
        teacher, pre, energy = prompt_teacher(model, controller, ids, c['prompt_tokens'])
        tracker = SequenceTracker(1, p.config.llm_layers, p.config.units, device='cuda', config=tracker_config(c))
        tracker.update(energy, torch.ones_like(energy, dtype=torch.bool), prompt=True)
        values.append(torch.quantile(tracker.scores()[:, 2:], 1-keep, dim=-1).mean(0))
        del teacher, pre
    return torch.stack(values).mean(0)


def calibration_sets(c):
    corpus, _, test = selections()
    _, rows = data('data/stage3-corpus-v1')
    validation = select_documents(rows, 'validation', c['validation_documents_per_domain'],
                                  {'seed': 20261111, 'minimum_sequence_length': 128})
    assert not {r['document_sha256'] for r in validation} & {r['document_sha256'] for r in test}
    assert len(test) == 63
    return corpus, validation, test


@torch.no_grad()
def calibrate(model, controller, p, q, offsets, corpus, validation, keep, root, c):
    output = root/'threshold.json'
    identity = {'target': keep, 'documents': validation, 'base_offsets': offsets.cpu().tolist(),
                'prompt_tokens': c['prompt_tokens'], 'tracker': asdict(tracker_config(c))}
    if output.exists():
        saved = json.loads(output.read_text()); assert saved['identity'] == identity
        return torch.tensor(saved['offsets'], device='cuda')
    trials = root/'threshold-trials.json'
    previous = json.loads(trials.read_text()) if trials.exists() else {'identity': identity, 'measurements': {}}
    assert previous['identity'] == identity
    def measure(shift):
        key = repr(float(shift))
        if key in previous['measurements']: return previous['measurements'][key]
        total = count = 0.
        for _, ids, lengths in pe.batches(corpus, validation, c['eval_batch_size']):
            value = decode_batch(model, controller, p, q, ids, lengths, offsets=offsets+shift, keep=keep,
                prompt_tokens=c['prompt_tokens'], tracker_config=tracker_config(c), quality=False, progress=progress)
            total += float(value['kept_sum'].sum()); count += float(value['sums'][:, 4].sum())
        used = total/count; previous['measurements'][key] = used; atomic_json(trials, previous)
        print(json.dumps({'stage': 'calibration', 'shift': shift, 'kept': used, 'target': keep}), flush=True)
        return used
    result = calibrate_threshold(measure, keep, tolerance=c['calibration_tolerance'],
                                 max_evaluations=c['calibration_max_evaluations'])
    result.update(identity=identity, offsets=(offsets+result['shift']).cpu().tolist())
    atomic_json(output, result)
    return offsets+result['shift']


@torch.no_grad()
def evaluate(model, controller, p, q, offsets, corpus, test, keep, root, c):
    output = root/'evaluation.json'
    if output.exists(): return json.loads(output.read_text())
    started = time.monotonic(); rows = []
    for metadata, ids, lengths in pe.batches(corpus, test, c['eval_batch_size']):
        r = decode_batch(model, controller, p, q, ids, lengths, offsets=offsets, keep=keep,
            prompt_tokens=c['prompt_tokens'], tracker_config=tracker_config(c), progress=progress)
        for i, meta in enumerate(metadata):
            a = r['sums'][i].tolist()
            row = dict(zip(('teacher_nll', 'student_nll', 'kl', 'top1_agreement', 'tokens'), a))
            row.update(document=meta['document_sha256'], domain=meta['domain'],
                kept_sum=float(r['kept_sum'][i]), kept_square=float(r['kept_square'][i]),
                per_layer_kept_sum=r['per_layer'][i].tolist(), reactivated=float(r['reactivated'][i]),
                reactivation_opportunities=float(r['reactivation_opportunities'][i]),
                never_selected_fraction=float(r['never_selected_fraction'][i]))
            rows.append(row)
        print(json.dumps({'stage': 'evaluation', 'documents': len(rows), 'seconds': time.monotonic()-started}), flush=True)
    def summarize(items):
        result = aggregate(items); n = result['tokens']
        mean = sum(r['kept_sum'] for r in items)/n
        result.update(routed_kept=mean, token_kept_std=math.sqrt(max(0.,sum(r['kept_square'] for r in items)/n-mean**2)),
            reactivation_rate=sum(r['reactivated'] for r in items)/max(1.,sum(r['reactivation_opportunities'] for r in items)),
            mean_document_never_selected_fraction=float(np.mean([r['never_selected_fraction'] for r in items])))
        return result
    overall = summarize(rows); overall.update(bootstrap_quality(rows, 2000, c['seed']))
    value = {'overall': overall, 'domains': {d: summarize([r for r in rows if r['domain']==d]) for d in ('web','code','conversation')},
        'per_layer_kept': (np.sum([r['per_layer_kept_sum'] for r in rows],axis=0)/overall['tokens']).tolist(),
        'prior_gain': p.prior_gain.detach().cpu().tolist(), 'prior_bias': p.prior_bias.detach().cpu().tolist(),
        'seconds': time.monotonic()-started, 'prompt_tokens': c['prompt_tokens'],
        'scoring': 'Input positions 64 through length-2 predict targets 65 through length-1; dense prompt excluded',
        'execution': 'Exact sequential cached student history, teacher-forced text, independent document trackers',
        'accounting': {**p.accounting(), 'new_parameters': 2*p.steps,
            'tracker_state_bytes_per_sequence': p.config.llm_layers*p.config.units*8,
            'sparse_kernel_speedup_measured': False}}
    atomic_json(root/'evaluation-documents.json', rows); atomic_json(output, value)
    return value


def train(c, config, variant, budget, smoke=False):
    root = ROOT/variant/f'budget-{budget}'; root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(c['seed'])
    p, offsets, source = load_planner(c, variant, budget)
    fp = fingerprint(config, variant, budget, source)
    if (root/'completion.json').exists() and not smoke:
        assert json.loads((root/'completion.json').read_text())['fingerprint'] == fp
        return
    model, controller = backbone(); q = load_file(str(CAL), device='cuda')['q75']
    if variant != 'baseline':
        corpus, records, calibration, order = training_data(c)
        new_params = [p.prior_gain, p.prior_bias] if p.prior_enabled else []
        base = [v for k,v in p.named_parameters() if v.requires_grad and k not in ('prior_gain','prior_bias')]
        groups = [{'params': base, 'multiplier': 1.}]
        if new_params: groups.append({'params': new_params, 'multiplier': c['prior_lr_multiplier'], 'weight_decay': 0.})
        optimizer = torch.optim.AdamW(groups, lr=c['quality_learning_rate'], weight_decay=c['weight_decay'])
        dual = DualBudget(budget/100, c['alpha_initial'], c['dual_learning_rate'], c['maximum_alpha'])
        seen = position = last_save = 0; elapsed = 0.
        if (root/'last.pt').exists() and not smoke:
            state = torch.load(root/'last.pt', map_location='cuda', weights_only=True)
            assert state['fingerprint'] == fp
            p.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            offsets=state['offsets']; dual=DualBudget(**state['dual']); seen=state['tokens']
            position=state['position']; elapsed=state['elapsed']; last_save=seen
            torch.set_rng_state(state['cpu_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu()); del state
        elif variant == 'prior_first':
            offsets = prior_offsets(model, controller, p, c, corpus, records, calibration, budget/100)
        total = 2048 if smoke else c['e2e_tokens_per_budget']
        assert total%c['sequence_length']==0 or total%c['sequence_length']>c['prompt_tokens']
        started=time.monotonic(); measured=0.; new_tokens=0
        torch.cuda.reset_peak_memory_stats()
        for index in range(position, len(order)):
            if seen>=total: break
            length=min(c['sequence_length'],total-seen)
            ids=pe.ids_for(corpus,records,order[index],length)
            factor=min(1.,(seen+length)/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/total)))
            for group in optimizer.param_groups:group['lr']=c['quality_learning_rate']*group['multiplier']*factor
            fraction=seen/total;mode='hard' if fraction>=1-c['hard_fraction'] else 'gumbel'
            temp=c['temperatures'][0]*(c['temperatures'][1]/c['temperatures'][0])**min(1.,fraction/(1-c['hard_fraction']))
            torch.cuda.synchronize();tick=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                metrics=end_to_end_batch(model,controller,p,optimizer,q,ids,offsets,dual,
                    prompt_tokens=c['prompt_tokens'],tracker_config=tracker_config(c),mode=mode,temperature=temp,
                    beta=c['beta'],epsilon=c['epsilon'],chunk_size=c['teacher_head_chunk_size'])
            torch.cuda.synchronize();measured+=time.monotonic()-tick;before=seen;seen+=length;new_tokens+=length
            info={'variant':variant,'budget':budget,'tokens':seen,'target_tokens':total,'position':index+1,
                'tokens_per_second':new_tokens/measured,'elapsed_seconds':elapsed+time.monotonic()-started,'metrics':metrics}
            if smoke or index%20==0 or seen==total:
                atomic_json(root/('smoke-progress.json' if smoke else 'progress.json'),info);print(json.dumps(info),flush=True)
                if not smoke:
                    with (root/'training-history.jsonl').open('a') as stream:stream.write(json.dumps(info)+'\n')
            if smoke:continue
            milestone=any(before<m<=seen for m in c['quality_checkpoint_tokens'])
            pause=pe.PAUSE or (ROOT/'PAUSE').exists()
            exceeded=elapsed+time.monotonic()-started>c['maximum_stage_hours']*3600
            if seen-last_save>=c['checkpoint_every_tokens'] or milestone or pause or exceeded:
                pe.save_resume(root,p,optimizer,offsets,dual,seen,index+1,elapsed+time.monotonic()-started,fp);last_save=seen
            if milestone:pe.save_weights(root,p,seen,fp)
            if pause:raise SystemExit(75)
            if exceeded:raise TimeoutError('Training time limit; resume checkpoint saved')
        assert seen==total
        info.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),fingerprint=fp,
            continuation_training_tokens=seen,scored_training_positions=seen-math.ceil(seen/c['sequence_length'])*c['prompt_tokens'],
            backbone_frozen=all(not v.requires_grad and v.grad is None for v in model.parameters()))
        atomic_json(root/('smoke.json' if smoke else 'training.json'),info)
        if smoke:return
        pe.save_weights(root,p,seen,fp)
        del optimizer;gc.collect();torch.cuda.empty_cache()
    p.eval();corpus,validation,test=calibration_sets(c)
    calibrated=calibrate(model,controller,p,q,offsets,corpus,validation,budget/100,root,c)
    evaluate(model,controller,p,q,calibrated,corpus,test,budget/100,root,c)
    atomic_json(root/'completion.json',{'status':'complete','fingerprint':fp,
        'extra_training_tokens':0 if variant=='baseline' else c['e2e_tokens_per_budget']})


def report(c):
    results={};comparisons={}
    rng=np.random.default_rng(c['seed'])
    for budget in c['budgets']:
        items={}
        for variant in ['baseline']+c['variants']:
            root=ROOT/variant/f'budget-{budget}'
            if not (root/'evaluation.json').exists():continue
            items[variant]=json.loads((root/'evaluation.json').read_text())
        results[str(budget)]=items
        for left,right in [('warm_prior','control'),('prior_first','control'),('warm_prior','baseline'),('prior_first','baseline')]:
            if left not in items or right not in items:continue
            a=json.loads((ROOT/left/f'budget-{budget}'/'evaluation-documents.json').read_text())
            b=json.loads((ROOT/right/f'budget-{budget}'/'evaluation-documents.json').read_text())
            assert [(r['document'],r['tokens']) for r in a]==[(r['document'],r['tokens']) for r in b]
            assert np.allclose([r['teacher_nll'] for r in a],[r['teacher_nll'] for r in b],rtol=1e-5)
            arrays=[np.array([r[k] for r in rows]) for rows,k in [(a,'student_nll'),(b,'student_nll'),(a,'teacher_nll'),(a,'tokens')]]
            indices=rng.integers(0,len(a),size=(2000,len(a)))
            aa,bb,dd,nn=[v[indices].sum(1) for v in arrays]
            delta=100*(np.exp((aa-dd)/nn)-np.exp((bb-dd)/nn))
            comparisons[f'{budget}:{left}-minus-{right}']={
                'ppl_increase_percentage_point_difference_95ci':np.quantile(delta,[.025,.975]).tolist(),
                'negative_favors_left':True,'method':'paired document bootstrap, 2000 draws'}
    value={'results':results,'comparisons':comparisons,'protocol':'Dense 64-token prompt; exact causal continuation evaluation',
        'no_comparison_to_historical_full_sequence_percentages':True}
    atomic_json(ROOT/'results.json',value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/sequence-prior-pilot.json'))
    parser.add_argument('--variant',choices=['baseline','warm_prior','prior_first','control']);parser.add_argument('--budget',type=int,choices=[50,40],default=50)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--report',action='store_true')
    args=parser.parse_args();c=json.loads(args.config.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGTERM,pe.request_pause);signal.signal(signal.SIGUSR1,pe.request_pause)
    if args.report:report(c)
    elif args.variant:train(c,args.config,args.variant,args.budget,args.smoke)
    else:parser.error('Choose variant or report')


if __name__=='__main__':main()
