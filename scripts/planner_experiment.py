"""Resumable copy, scheduled-feedback and output-quality planner experiments.

Every stage has its own token counter and checkpoint. Evaluation is actual
sequential decoding, batching independent documents without sharing histories.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.open_loop import PlannerConfig, SuffixPlanner
from pathpred.planner_training import copy_batch, end_to_end_batch, copy_targets
from pathpred.planner_route import record_parallel_paths
from pathpred.planner_decode import decode_batch
from pathpred.end_to_end import DualBudget
from pathpred.budget_calibration import calibrate_threshold
from scripts.neuron_guessing import backbone, data, CAL
from scripts.calibrate_native import selections
from scripts.oracle_quality import select_documents, aggregate, bootstrap_quality

ROOT = Path('artifacts/planner-v1')
VARIANTS = ('autoregressive', 'one_shot_history', 'one_shot_hidden')
PAUSE = False


def request_pause(signum, frame):
    global PAUSE
    PAUSE = True


def decode_progress(token, total):
    if PAUSE or (ROOT/'PAUSE').exists(): raise SystemExit(75)
    print(json.dumps({'stage': 'cached_decode_progress', 'position': token, 'positions': total}), flush=True)


def planner(variant, c, device='cuda'):
    if variant not in VARIANTS: raise ValueError(variant)
    return SuffixPlanner(PlannerConfig(width=c['width'], depth=c['depth'], heads=c['heads'],
        autoregressive=variant == 'autoregressive', previous_path=variant != 'one_shot_hidden')).to(device)


def fingerprint(config, variant):
    c = json.loads(config.read_text())
    files = list(Path('pathpred').glob('*.py')) + [config, Path('scripts/planner_experiment.py'),
        Path('scripts/neuron_guessing.py'), Path('scripts/calibrate_native.py'),
        Path('scripts/oracle_quality.py'), Path(c['copy_corpus'])/'manifest.json',
        Path(c['quality_corpus'])/'manifest.json', Path('data/stage3-corpus-v1/manifest.json'), CAL]
    return {'variant': variant, 'files': {str(p): file_sha256(p) for p in sorted(set(files))}}


def ordered_data(c, quality=False):
    source = c['quality_corpus'] if quality else c['copy_corpus']
    corpus, records = data(source)
    _, heldout = data('data/stage3-corpus-v1')
    forbidden = {r['document_sha256'] for r in heldout if r['split'] != 'train'}
    order = [i for i, r in enumerate(records) if r['split'] == 'train' and
             r['length'] >= c['sequence_length'] and r['document_sha256'] not in forbidden]
    order.sort(key=lambda i: hashlib.sha256((str(c.get('data_seed',c['seed'])) + ':' +
        records[i]['document_sha256'] + ':' + str(records[i]['document_token_offset'])).encode()).hexdigest())
    calibration, order = order[:8], order[8:]
    required = c['e2e_tokens_per_budget'] if quality else c['copy_tokens'] + c['scheduled_tokens'] + 512
    assert len(order)*c['sequence_length'] >= required, 'Insufficient distinct full training windows'
    audit = {'source': source, 'training_only': True, 'validation_test_hashes_excluded': True,
        'available_full_windows': len(order), 'reserved_calibration_windows': calibration,
        'order_sha256': hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        'same_order_all_variants': True, 'distinct_windows': len(set(order)),
        'note': 'New architectures reuse prior training corpus for copying; quality corpus is fresh relative to that corpus.'}
    atomic_json(ROOT/('quality-data-audit.json' if quality else 'copy-data-audit.json'), audit)
    return corpus['input_ids'], records, calibration, order


def ids_for(corpus, records, index, length=512):
    row = records[index]
    return corpus[row['sequence_id']:row['sequence_id']+1, :length].long().cuda()


def save_resume(root, p, optimizer, offsets, dual, tokens, position, elapsed, fp):
    value = {'model': p.state_dict(), 'optimizer': optimizer.state_dict(),
        'offsets': offsets, 'dual': None if dual is None else asdict(dual),
        'tokens': tokens, 'position': position, 'elapsed': elapsed, 'fingerprint': fp,
        'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}
    root.mkdir(parents=True, exist_ok=True)
    temporary = root/'last.tmp'; torch.save(value, temporary); temporary.replace(root/'last.pt')


def save_weights(root, p, tokens, fp):
    target = root/f'planner-{tokens:08d}.safetensors'
    meta = root/f'checkpoint-{tokens:08d}.json'
    if meta.exists():
        old = json.loads(meta.read_text())
        assert old['fingerprint'] == fp and file_sha256(target) == old['file']['sha256']
        return old
    entry = atomic_tensors(target, {k: v.detach().cpu().contiguous() for k, v in p.state_dict().items()})
    value = {'tokens_in_this_stage': tokens, 'file': entry, 'fingerprint': fp, 'accounting': p.accounting()}
    atomic_json(meta, value); return value


def load_stage_weights(p, root, tokens):
    meta = json.loads((root/f'checkpoint-{tokens:08d}.json').read_text())
    path = root/meta['file']['file']; assert file_sha256(path) == meta['file']['sha256']
    p.load_state_dict(load_file(str(path), device='cuda'))


@torch.no_grad()
def training_offsets(model, controller, p, q, corpus, records, windows, keep):
    values = []
    p.eval()
    for index in windows:
        ids = ids_for(corpus, records, index)
        hidden, _, _, _ = copy_targets(model, controller, q, ids, keep)
        previous = valid = None
        if p.config.previous_path:
            previous, valid = record_parallel_paths(model, controller, p, q, ids, keep=keep)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = p.plan(hidden, previous, valid, mode='topk', keep=keep)
        values.append(torch.quantile(result.scores.float(), 1-keep, dim=-1).mean((0, 1)))
    return torch.stack(values).mean(0)


def batches(corpus, selected, batch_size):
    for start in range(0, len(selected), batch_size):
        rows = selected[start:start+batch_size]
        lengths = torch.tensor([min(r['length'], 512) for r in rows], device='cuda')
        # Identical document order, padding, and partition across all methods.
        ids = torch.zeros(len(rows), int(lengths.max()), dtype=torch.long, device='cuda')
        for i, r in enumerate(rows):
            ids[i, :lengths[i]] = corpus[r['sequence_id'], :lengths[i]].long().cuda()
        yield rows, ids, lengths


@torch.no_grad()
def calibrate(model, controller, p, q, offsets, corpus, validation, keep, output, c,
              drop_previous=False, closed_loop=False):
    identity = {'target': keep, 'validation_documents': validation, 'base_offsets': offsets.cpu().tolist(),
        'execution': 'exact sequential language-token history; independent documents batched',
        'batch_size': c['eval_batch_size'], 'drop_previous': drop_previous, 'closed_loop': closed_loop,
        'tolerance': c['calibration_tolerance'], 'max_evaluations': c['calibration_max_evaluations']}
    if output.exists():
        result = json.loads(output.read_text())
        assert result['identity'] == identity
        return torch.tensor(result['offsets'], device='cuda')
    trials = output.with_name(output.stem+'-trials.json')
    previous = json.loads(trials.read_text()) if trials.exists() else {'identity': identity, 'measurements': {}}
    if previous['identity'] != identity and c.get('reuse_calibration_trials',False):
        omit = {'tolerance','max_evaluations'}
        assert {k:v for k,v in previous['identity'].items() if k not in omit} == {k:v for k,v in identity.items() if k not in omit}
        previous['inherited_identity'] = previous['identity']
        previous['identity'] = identity
    assert previous['identity'] == identity
    def measure(shift):
        key = repr(float(shift))
        if key in previous['measurements']: return previous['measurements'][key]
        total = count = 0.
        for _, ids, lengths in batches(corpus, validation, c['eval_batch_size']):
            result = decode_batch(model, controller, p, q, ids, lengths, offsets=offsets+shift,
                keep=keep, quality=False, drop_previous=drop_previous, closed_loop=closed_loop,
                progress=decode_progress)
            total += float(result['kept_sum'].sum()); count += float(result['sums'][:, 4].sum())
        used = total/count
        previous['measurements'][key] = used; atomic_json(trials, previous)
        print(json.dumps({'stage': 'exact_history_threshold_search', 'shift': shift,
                          'kept': used, 'target': keep}), flush=True)
        return used
    result = calibrate_threshold(measure, keep, tolerance=c['calibration_tolerance'],
                                 max_evaluations=c['calibration_max_evaluations'],strict=c.get('calibration_strict',True))
    result.update(identity=identity, offsets=(offsets+result['shift']).cpu().tolist())
    atomic_json(output, result)
    return offsets+result['shift']


@torch.no_grad()
def evaluate(model, controller, p, q, offsets, corpus, selected, keep, output, c,
             drop_previous=False, closed_loop=False):
    if output.exists(): return json.loads(output.read_text())
    started = time.monotonic(); rows = []; all_counts = []
    for metadata, ids, lengths in batches(corpus, selected, c['eval_batch_size']):
        result = decode_batch(model, controller, p, q, ids, lengths, offsets=offsets,
            keep=keep, drop_previous=drop_previous, closed_loop=closed_loop, progress=decode_progress)
        for i, meta in enumerate(metadata):
            a = result['sums'][i].tolist(); counts = result['layer_counts'][i]
            row = dict(zip(('teacher_nll', 'student_nll', 'kl', 'top1_agreement', 'tokens'), a))
            row.update(document=meta['document_sha256'], domain=meta['domain'],
                       routed_kept=float(result['kept_sum'][i]/a[4]), layer_counts=counts.tolist())
            rows.append(row); all_counts.append(counts)
        print(json.dumps({'stage': 'exact_history_evaluation', 'documents': len(rows),
                          'total': len(selected), 'seconds': time.monotonic()-started}), flush=True)
    overall = aggregate(rows); overall.update(bootstrap_quality(rows, 2000, c['seed']))
    overall['routed_kept'] = sum(r['routed_kept']*r['tokens'] for r in rows)/overall['tokens']
    domains = {}
    for name in ('web', 'code', 'conversation'):
        subset = [r for r in rows if r['domain'] == name]; values = aggregate(subset)
        values['routed_kept'] = sum(r['routed_kept']*r['tokens'] for r in subset)/values['tokens']; domains[name] = values
    counts = torch.stack(all_counts).sum(0); by_layer = []
    for i, (tp, positive, selected_count, union) in enumerate(counts.tolist()):
        by_layer.append({'layer_one_based': i+3, 'recall': tp/positive,
            'precision': tp/selected_count if selected_count else None, 'iou': tp/union if union else None,
            'reference': 'dense full-model magnitude top-k, not a global optimum'})
    value = {'overall': overall, 'domains': domains, 'per_layer_oracle_accuracy': by_layer,
        'offsets': offsets.cpu().tolist(), 'drop_previous': drop_previous,
        'execution': 'exact sequential history with independent-document batching',
        'batch_size': c['eval_batch_size'], 'seconds': time.monotonic()-started,
        'historical_full_sequence_numbers_not_directly_substituted': True}
    atomic_json(output.with_name(output.stem+'-documents.json'), rows); atomic_json(output, value)
    return value


def train(c, config_path, variant, stage, budget):
    fp = fingerprint(config_path, variant); torch.manual_seed(c['seed'])
    is_quality = stage == 'e2e'
    root = ROOT/variant/(f'budget-{budget}' if is_quality else stage); root.mkdir(parents=True, exist_ok=True)
    if (root/'completion.json').exists():
        assert json.loads((root/'completion.json').read_text())['fingerprint'] == fp
        print('Stage already complete', flush=True); return
    corpus, records, calibration, order = ordered_data(c, quality=is_quality)
    if stage == 'scheduled': order = order[math.ceil(c['copy_tokens']/512):]
    total = c['e2e_tokens_per_budget'] if is_quality else c[stage+'_tokens']
    model, controller = backbone(); q = load_file(str(CAL), device='cuda')['q75']
    p = planner(variant, c)
    if stage != 'copy':
        prior = 'scheduled' if is_quality else 'copy'
        load_stage_weights(p, ROOT/variant/prior, c[prior+'_tokens'])
    elif CAL.exists():
        cal = load_file(str(CAL), device='cuda'); prior = ((cal['frequency_50']+cal['frequency_40'])/2)[2:].clamp(.001,.999)
        with torch.no_grad():
            p.output_bias.copy_((c['positive_weight']*prior/(1-prior)).log()); p.output_weight.mul_(.01)
    lr = c['quality_learning_rate'] if is_quality else c['copy_learning_rate']
    optimizer = torch.optim.AdamW(p.parameters(), lr=lr, weight_decay=c['weight_decay'])
    offsets = torch.zeros(p.steps, device='cuda'); dual = None
    seen = position = last_save = 0; elapsed = 0.
    if is_quality: dual = DualBudget(budget/100, c['alpha_initial'], c['dual_learning_rate'], c['maximum_alpha'])
    if (root/'last.pt').exists():
        saved = torch.load(root/'last.pt', map_location='cuda', weights_only=True); assert saved['fingerprint'] == fp
        p.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); offsets = saved['offsets']
        if is_quality: dual = DualBudget(**saved['dual'])
        seen = saved['tokens']; position = saved['position']; elapsed = saved['elapsed']; last_save = seen
        torch.set_rng_state(saved['cpu_rng'].cpu()); torch.cuda.set_rng_state(saved['cuda_rng'].cpu()); del saved
    elif is_quality:
        offsets = training_offsets(model, controller, p, q, corpus, records, calibration, budget/100)
        atomic_json(root/'initial-offsets.json', offsets.cpu().tolist())
    started = time.monotonic(); train_seconds = 0.; new_tokens = 0
    milestones = c['quality_checkpoint_tokens'] if is_quality else [total]
    torch.cuda.reset_peak_memory_stats()
    for index in range(position, len(order)):
        if seen >= total: break
        length = min(512, min(n for n in milestones if n > seen)-seen)
        ids = ids_for(corpus, records, order[index], length)
        factor = min(1., (seen+length)/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/total)))
        for group in optimizer.param_groups: group['lr'] = lr*factor
        torch.cuda.synchronize(); tick = time.monotonic()
        if is_quality:
            fraction = seen/total; mode = 'hard' if fraction >= 1-c['hard_fraction'] else 'gumbel'
            temperature = c['temperatures'][0]*(c['temperatures'][1]/c['temperatures'][0])**min(1., fraction/(1-c['hard_fraction']))
            metrics = end_to_end_batch(model, controller, p, optimizer, q, ids, offsets, dual,
                mode=mode, temperature=temperature, beta=c['beta'], epsilon=c['epsilon'],
                chunk_size=c['teacher_head_chunk_size'])
        else:
            metrics = copy_batch(model, controller, p, optimizer, q, ids, keep=(.5 if index%2==0 else .4),
                scheduled=stage=='scheduled', teacher_probability=max(0.,1-seen/total), positive_weight=c['positive_weight'])
        torch.cuda.synchronize(); train_seconds += time.monotonic()-tick; seen += length; new_tokens += length
        with (root/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(tokens=seen,batch_start_tokens=seen-length,batch_tokens=length,stage=stage,variant=variant,budget=budget if is_quality else None,learning_rate=optimizer.param_groups[0]['lr'],gate_mode=mode if is_quality else 'topk' if stage=='scheduled' else 'teacher_forced',temperature=temperature if is_quality else None,metrics=metrics))+'\n')
        if index%20 == 0 or seen in milestones:
            progress = {'variant': variant, 'stage': stage, 'budget': budget if is_quality else None,
                'tokens_in_stage': seen, 'target_tokens': total, 'position': index+1,
                'tokens_per_second': new_tokens/train_seconds, 'elapsed_seconds': elapsed+time.monotonic()-started,
                'metrics': metrics}
            atomic_json(root/'progress.json', progress); print(json.dumps(progress), flush=True)
        exceeded = elapsed+time.monotonic()-started > c['maximum_stage_hours']*3600
        pause = PAUSE or (ROOT/'PAUSE').exists()
        if seen-last_save >= c['checkpoint_every_tokens'] or seen in milestones or pause or exceeded:
            save_resume(root, p, optimizer, offsets, dual, seen, index+1, elapsed+time.monotonic()-started, fp); last_save=seen
        if seen in milestones: save_weights(root, p, seen, fp)
        if pause:
            atomic_json(root/'paused.json', {'tokens': seen, 'fingerprint': fp}); raise SystemExit(75)
        if exceeded: raise TimeoutError('Stage runtime guard: resumable state saved')
    assert seen == total
    save_weights(root, p, seen, fp)
    atomic_json(root/'training.json', {'status': 'complete', 'tokens_in_stage': seen, 'fingerprint': fp,
        'stage': stage, 'variant': variant, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'seconds': elapsed+time.monotonic()-started, 'accounting': p.accounting(),
        'backbone_frozen': all(not v.requires_grad and v.grad is None for v in model.parameters())})
    # Release Adam state before expensive sequential validation and evaluation.
    del optimizer; gc.collect(); torch.cuda.empty_cache(); p.eval()
    if is_quality:
        eval_corpus, _, test = selections()
        _, all_meta = data('data/stage3-corpus-v1')
        validation = select_documents(all_meta, 'validation', c['validation_documents_per_domain'],
                                      {'seed': 20261111, 'minimum_sequence_length': 128})
        assert not {r['document_sha256'] for r in validation}&{r['document_sha256'] for r in test}
        calibrated = calibrate(model, controller, p, q, offsets, eval_corpus, validation,
                              budget/100, root/'threshold.json', c)
        evaluate(model, controller, p, q, calibrated, eval_corpus, test, budget/100, root/'evaluation.json', c)
        if c.get('run_previous_path_ablation',True) and p.config.previous_path and variant == 'autoregressive':
            evaluate(model, controller, p, q, calibrated, eval_corpus, test, budget/100,
                     root/'no-previous-fixed.json', c, drop_previous=True)
            no_previous = calibrate(model, controller, p, q, offsets, eval_corpus, validation,
                                   budget/100, root/'no-previous-threshold.json', c, drop_previous=True)
            evaluate(model, controller, p, q, no_previous, eval_corpus, test, budget/100,
                     root/'no-previous-matched.json', c, drop_previous=True)
    atomic_json(root/'completion.json', {'status': 'complete', 'fingerprint': fp, 'tokens': seen})


def smoke(c):
    model, controller = backbone(); q = load_file(str(CAL), device='cuda')['q75']
    corpus, records = data(c['copy_corpus'])
    chosen = next(r for r in records if r['split']=='train' and r['length']>=512)
    ids = corpus['input_ids'][chosen['sequence_id']:chosen['sequence_id']+1,:512].long().cuda()
    results = {}
    for variant in c['variants']:
        torch.manual_seed(c['seed']); p = planner(variant,c); optimizer=torch.optim.AdamW(p.parameters(),lr=c['quality_learning_rate'])
        torch.cuda.reset_peak_memory_stats(); timings={}
        for stage in ('copy','scheduled','e2e'):
            torch.cuda.synchronize();tick=time.monotonic()
            if stage=='e2e':
                metrics=end_to_end_batch(model,controller,p,optimizer,q,ids,torch.zeros(p.steps,device='cuda'),
                    DualBudget(.5,.1,.05,100.),chunk_size=c['teacher_head_chunk_size'])
            else:
                metrics=copy_batch(model,controller,p,optimizer,q,ids,keep=.5,scheduled=stage=='scheduled',teacher_probability=.5)
            torch.cuda.synchronize();seconds=time.monotonic()-tick
            timings[stage]={'seconds':seconds,'tokens_per_second':ids.numel()/seconds,'metrics':metrics}
        del optimizer; gc.collect();torch.cuda.empty_cache();p.eval()
        short=ids[:,:32].expand(4,-1).contiguous();lengths=torch.full((4,),32,device='cuda')
        tick=time.monotonic();z=decode_batch(model,controller,p,q,short,lengths,offsets=torch.zeros(p.steps,device='cuda'),keep=.5)
        torch.cuda.synchronize();seconds=time.monotonic()-tick
        results[variant]={'training':timings,'decode_probe':{'documents':4,'sequence_length':32,
            'seconds':seconds,'scored_tokens':float(z['sums'][:,4].sum()),
            'caution':'Short-context timing does not establish 512-token exact-calibration cost'},
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'accounting':p.accounting()}
        atomic_json(ROOT/'gpu-smoke.json',results);print(json.dumps({'smoke':variant,**results[variant]}),flush=True)
        del p;gc.collect();torch.cuda.empty_cache()


def reference(c, config_path, name, budget):
    from pathpred.neuron_prediction import NeuronPredictor, neuron_config
    source = Path('artifacts')/('e2e-full-v1' if name=='binary' else 'e2e-hidden-full-v1')/f'budget-{budget}'
    metadata = json.loads((source/'checkpoint-10000000.json').read_text())
    weight = source/metadata['file']['file']; assert file_sha256(weight)==metadata['file']['sha256']
    root=ROOT/'references'/name/f'budget-{budget}';root.mkdir(parents=True,exist_ok=True)
    model,controller=backbone();q=load_file(str(CAL),device='cuda')['q75']
    p=NeuronPredictor(neuron_config(name)).cuda().eval();p.load_state_dict(load_file(str(weight),device='cuda'))
    offsets=torch.tensor(json.loads((source/'threshold-10000000.json').read_text())['offsets'],device='cuda')
    corpus,_,test=selections();_,rows=data('data/stage3-corpus-v1')
    validation=select_documents(rows,'validation',c['validation_documents_per_domain'],
                                {'seed':20261111,'minimum_sequence_length':128})
    calibrated=calibrate(model,controller,p,q,offsets,corpus,validation,budget/100,root/'threshold.json',c,closed_loop=True)
    evaluate(model,controller,p,q,calibrated,corpus,test,budget/100,root/'evaluation.json',c,closed_loop=True)
    atomic_json(root/'completion.json',{'status':'complete','source':str(weight),'source_sha256':metadata['file']['sha256'],
        'new_training_tokens':0,'fingerprint':fingerprint(config_path,'reference-'+name),
        'note':'Existing closed-loop weights, recalibrated and evaluated under the exact cached protocol. Training histories differ from the new planner pilot.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/planner-pilot.json'))
    parser.add_argument('--variant',choices=VARIANTS);parser.add_argument('--stage',choices=['copy','scheduled','e2e'])
    parser.add_argument('--budget',type=int,choices=[50,40],default=50);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--reference',choices=['binary','hidden'])
    args=parser.parse_args();c=json.loads(args.config.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGUSR1,request_pause)
    if args.smoke:smoke(c)
    elif args.reference:reference(c,args.config,args.reference,args.budget)
    elif args.variant and args.stage:train(c,args.config,args.variant,args.stage,args.budget)
    else:parser.error('Choose --smoke or --variant and --stage')


if __name__=='__main__':main()
