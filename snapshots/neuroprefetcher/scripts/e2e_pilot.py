"""Warm-started end-to-end neuron gating: pilot, train, hard evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.neuron_prediction import NeuronPredictor, neuron_config
from pathpred.end_to_end import quality_route, dense_teacher, quality_loss, DualBudget
from pathpred.oracle_checks import quality_rows
from scripts.neuron_guessing import backbone, data, training_order, CAL
from scripts.oracle_quality import aggregate, bootstrap_quality, select_documents

ROOT = Path('artifacts/e2e-pilot-v1')
WARM = Path('artifacts/neuron-guessing-v1/binary.safetensors')


def config():
    return json.loads(Path('configs/e2e-pilot.json').read_text())


def source_fingerprint():
    names = ['configs/e2e-pilot.json', 'docs/e2e-pilot-protocol.md',
             'scripts/e2e_pilot.py', 'pathpred/end_to_end.py',
             'pathpred/neuron_prediction.py', 'pathpred/path_model.py',
             'pathpred/gated_qwen.py', 'pathpred/reordering.py',
             'data/neuron-training-corpus-v1/manifest.json',
             'data/stage3-corpus-v1/manifest.json']
    return {name: file_sha256(Path(name)) for name in names}


def predictor():
    assert file_sha256(WARM) == config()['warm_start_sha256']
    p = NeuronPredictor(neuron_config('binary')).cuda().eval()
    p.load_state_dict(load_file(str(WARM), device='cuda'))
    return p


def training_data():
    tensors, records = data('data/neuron-training-corpus-v1')
    _, heldout = data('data/stage3-corpus-v1')
    previous = json.loads(Path('artifacts/neuron-guessing-v1/training.json').read_text())
    old_order = training_order(records)
    used = set(old_order[:previous['distinct_windows']])
    forbidden = {r['document_sha256'] for r in heldout if r['split'] != 'train'}
    available = [i for i in old_order[previous['distinct_windows']:]
                 if records[i]['length'] >= config()['sequence_length']]
    assert not used.intersection(available)
    assert all(records[i]['split'] == 'train' and records[i]['document_sha256'] not in forbidden for i in available)
    # Separate training-only calibration windows from gradient updates.
    calibration, order = available[:4], available[4:]
    assert len(order) * config()['sequence_length'] >= config()['train_tokens']
    atomic_json(ROOT/'data-audit.json', {'available_full_windows': len(order),
        'calibration_windows': calibration, 'warm_start_training_windows_excluded': len(used),
        'no_repeated_windows': len(order) == len(set(order)),
        'validation_test_document_hashes_disjoint': True,
        'note': 'New windows, not necessarily new documents, relative to imitation training.'})
    return tensors['input_ids'], records, calibration, order


@torch.no_grad()
def calibrate_offsets(model, controller, p, q, corpus, records, windows, keep):
    values = []
    for i in windows:
        ids = corpus[records[i]['sequence_id']:records[i]['sequence_id']+1, :512].long().cuda()
        with quality_route(model, controller, p, q, mode='topk', keep=keep, checkpoint_layers=False) as trace:
            model.model(ids, use_cache=False)
        values.append(torch.stack([torch.quantile(trace['scores'][l].float(), 1-keep, dim=-1).mean()
                                   for l in range(2, 28)]))
    offset = torch.cat((torch.zeros(2, device='cuda'), torch.stack(values).mean(0)))
    return offset


def schedule(tokens):
    c = config(); phase = 0 if tokens < c['phase_50_tokens'] else 1
    start = 0 if phase == 0 else c['phase_50_tokens']
    length = c['phase_50_tokens'] if phase == 0 else c['train_tokens']-start
    fraction = (tokens-start)/length
    temp = c['temperatures'][0] * (c['temperatures'][1]/c['temperatures'][0]) ** min(1., fraction/(1-c['hard_fraction']))
    mode = 'hard' if fraction >= 1-c['hard_fraction'] else 'soft'
    return phase, (.5 if phase == 0 else .4), temp, mode


def train_batch(model, controller, p, optimizer, q, offsets, ids, dual, temperature, mode, lr):
    c = config(); optimizer.zero_grad(set_to_none=True)
    target, teacher_pre = dense_teacher(model, controller, ids)
    for group in optimizer.param_groups: group['lr'] = lr * group.get('lr_multiplier', 1.)
    with quality_route(model, controller, p, q, temperature=temperature, mode=mode,
                       score_offsets=offsets) as trace:
        hidden = model.model(ids, use_cache=False).last_hidden_state
        loss, metrics = quality_loss(model, hidden, trace['pre_norm'], target, teacher_pre, trace,
            alpha=dual.alpha, beta=c['beta'], epsilon=c['epsilon'],
            chunk_size=c['teacher_head_chunk_size'])
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite end-to-end loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(p.parameters(), 1., error_if_nonfinite=True)
        assert trace['predictor_calls'] == p.steps
        assert all(v.grad is None and not v.requires_grad for v in model.parameters())
        values = {name: float(value.detach()) for name, value in metrics.items()}
        values.update(loss=float(loss.detach()), gradient_norm=float(gradient),
                      recomputations=trace['recomputations'], alpha=dual.alpha,
                      all_mlp_used=(2+26*values['used'])/28)
        optimizer.step()
    dual.update(values['used'])
    return values


def checkpoint_state(p, optimizer, offsets, dual, tokens, position, elapsed, phase, fp):
    path = ROOT/'last.pt'; temporary = path.with_suffix('.tmp')
    torch.save({'predictor': p.state_dict(), 'optimizer': optimizer.state_dict(),
                'offsets': offsets, 'dual': asdict(dual), 'tokens': tokens,
                'next_position': position, 'elapsed_seconds': elapsed,
                'phase': phase, 'fingerprint': fp,
                'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate_checkpoint(model, controller, p, q, offsets, label, trained_tokens):
    """Always report both exact budgets and the native threshold gate policy."""
    c = config(); corpus, records = data('data/stage3-corpus-v1')
    selection_config = {'seed': 20261109, 'minimum_sequence_length': 128}
    selected = select_documents(records, 'test', 24, selection_config)
    rows = []; started = time.monotonic()
    for i, meta in enumerate(selected):
        length = min(meta['length'], 512)
        ids = corpus['input_ids'][meta['sequence_id']:meta['sequence_id']+1, :length].long().cuda()
        teacher, _ = dense_teacher(model, controller, ids)
        conditions = {}
        for name, mode, keep in [('topk-50', 'topk', .5), ('topk-40', 'topk', .4), ('native', 'hard', .5)]:
            with quality_route(model, controller, p, q, mode=mode, keep=keep,
                               score_offsets=offsets, checkpoint_layers=False) as trace:
                student = model(ids, use_cache=False).logits
            masks = torch.stack([trace['gates'][l] for l in range(2,28)], 2)
            assert ((masks == 0) | (masks == 1)).all()
            if mode == 'topk': assert (masks.sum(-1) == math.ceil(6144*keep)).all()
            for l in range(28):
                mask = trace['gates'][l].detach() >= .5
                assert (trace['bits'][l][~mask] == -1).all()
                if l < 2: assert mask.all()
            metrics = quality_rows(teacher, student, ids)
            row = {k: float(v.double().sum()) for k,v in metrics.items()}
            per_token = masks.float().mean((2,3))
            row.update(tokens=length-1, domain=meta['domain'], document=meta['document_sha256'],
                       routed_kept=float(per_token[:,:-1].mean()),
                       routed_kept_min=float(per_token.min()), routed_kept_max=float(per_token.max()))
            conditions[name] = row
        rows.append({'metadata': meta, 'conditions': conditions})
        if (i+1) % 10 == 0:
            print(json.dumps({'stage': 'evaluation', 'checkpoint': label, 'documents': i+1, 'total': len(selected)}), flush=True)
    result = {'checkpoint': label, 'trained_tokens': trained_tokens,
              'documents': len(rows), 'seconds': time.monotonic()-started, 'conditions': {}}
    for name in rows[0]['conditions']:
        values = [r['conditions'][name] for r in rows]; overall = aggregate(values)
        overall.update(bootstrap_quality(values, 2000, c['seed']))
        overall['routed_kept'] = sum(r['routed_kept']*r['tokens'] for r in values)/overall['tokens']
        overall['all_mlp_kept'] = (2+26*overall['routed_kept'])/28
        result['conditions'][name] = {'overall': overall,
            'domains': {d: aggregate([r for r in values if r['domain']==d]) for d in ['web','code','conversation']}}
    atomic_json(ROOT/f'evaluation-{label}.json', result)
    atomic_json(ROOT/f'documents-{label}.json', rows)
    return result


def run(pilot=False):
    c = config(); fp = source_fingerprint(); torch.manual_seed(c['seed'])
    corpus, records, calibration, order = training_data()
    model, controller = backbone(); p = predictor(); q = load_file(str(CAL), device='cuda')['q75']
    offsets = calibrate_offsets(model, controller, p, q, corpus, records, calibration, .5)
    optimizer = torch.optim.AdamW(p.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    dual = DualBudget(.5, c['alpha_initial'], c['dual_learning_rate'], c['maximum_alpha'])
    seen = position = phase = 0; elapsed = 0.; last_save = 0
    if not pilot and (ROOT/'last.pt').exists():
        saved = torch.load(ROOT/'last.pt', map_location='cuda', weights_only=True)
        assert saved['fingerprint'] == fp
        p.load_state_dict(saved['predictor']); optimizer.load_state_dict(saved['optimizer'])
        offsets = saved['offsets']; dual = DualBudget(**saved['dual'])
        seen = saved['tokens']; position = saved['next_position']; phase = saved['phase']
        elapsed = saved['elapsed_seconds']; last_save = seen
        torch.set_rng_state(saved['cpu_rng'].cpu()); torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        del saved
    if not pilot and seen == 0:
        atomic_json(ROOT/'initial-offsets.json', offsets.cpu().tolist())
        evaluate_checkpoint(model, controller, p, q, offsets, '0000000', 0)
    if not pilot and seen in c['milestones'] and not (ROOT/f'evaluation-{seen:07d}.json').exists():
        # A process can stop after the atomic resume save but before its
        # checkpoint evaluation. Finish that milestone before advancing.
        label = f'{seen:07d}'
        entry = atomic_tensors(ROOT/f'predictor-{label}.safetensors',
            {k:v.detach().cpu().contiguous() for k,v in p.state_dict().items()})
        atomic_json(ROOT/f'checkpoint-{label}.json', {'tokens':seen,
            'phase_target':dual.target, 'mode':schedule(seen-1)[3],
            'offsets':offsets.cpu().tolist(), 'dual':asdict(dual), 'file':entry})
        evaluate_checkpoint(model, controller, p, q, offsets, label, seen)
    started = time.monotonic(); durations = []; presented = 0; update_windows = position
    torch.cuda.reset_peak_memory_stats()
    for j in range(position, len(order)):
        if seen >= c['train_tokens']: break
        new_phase, keep, temperature, mode = schedule(seen)
        if new_phase != phase:
            # Use only reserved training windows, never evaluation quality, for
            # the operating-point change from 50% to 40%.
            offsets = calibrate_offsets(model, controller, p, q, corpus, records, calibration, keep)
            dual = DualBudget(keep, c['alpha_initial'], c['dual_learning_rate'], c['maximum_alpha'])
            phase = new_phase
            atomic_json(ROOT/'phase-40-offsets.json', offsets.cpu().tolist())
        next_milestone = min(x for x in c['milestones'] if x > seen)
        length = min(512, next_milestone-seen)
        meta = records[order[j]]
        ids = corpus[meta['sequence_id']:meta['sequence_id']+1, :length].long().cuda()
        factor = min(1., (seen+length)/c['warmup_tokens']) * (.1+.9*.5*(1+math.cos(math.pi*seen/c['train_tokens'])))
        # The short pilot includes soft and hard backward paths at full LR.
        if pilot: mode = 'soft' if len(durations) < 4 else 'hard'; factor = 1.
        torch.cuda.synchronize(); tick = time.monotonic()
        metrics = train_batch(model, controller, p, optimizer, q, offsets, ids,
                              dual, temperature, mode, c['learning_rate']*factor)
        torch.cuda.synchronize(); duration = time.monotonic()-tick
        durations.append(duration); presented += length; seen += length; update_windows = j+1
        total_elapsed = elapsed+time.monotonic()-started
        progress = {'tokens': seen, 'phase_target': keep, 'mode': mode, 'temperature': temperature,
                    'elapsed_seconds': total_elapsed, 'tokens_per_second': presented/sum(durations),
                    'step_seconds': duration, 'metrics': metrics}
        if pilot or j % 20 == 0:
            atomic_json(ROOT/('pilot-progress.json' if pilot else 'progress.json'), progress)
            print(json.dumps(progress), flush=True)
        if pilot and len(durations) == 8:
            atomic_json(ROOT/'pilot.json', {'status': 'complete', 'tokens': presented,
                'seconds': sum(durations), 'tokens_per_second': presented/sum(durations),
                'estimated_3m_hours': c['train_tokens']/(presented/sum(durations))/3600,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                'steps': durations, 'last_metrics': metrics, 'fingerprint': fp,
                'scope': 'Dense teacher + differentiable student + full-vocabulary KL + predictor update; 4 soft and 4 hard steps. Pilot updates discarded.'})
            return
        guard = total_elapsed > c['maximum_training_hours']*3600
        if not pilot and (seen-last_save >= c['checkpoint_every_tokens'] or seen in c['milestones'] or guard):
            checkpoint_state(p, optimizer, offsets, dual, seen, j+1, total_elapsed, phase, fp)
            last_save = seen
        if guard: raise TimeoutError('Four-hour training guard: resumable checkpoint saved')
        if seen in c['milestones']:
            label = f'{seen:07d}'
            entry = atomic_tensors(ROOT/f'predictor-{label}.safetensors',
                {k:v.detach().cpu().contiguous() for k,v in p.state_dict().items()})
            atomic_json(ROOT/f'checkpoint-{label}.json', {'tokens':seen,'phase_target':keep,
                'mode':mode,'offsets':offsets.cpu().tolist(),'dual':asdict(dual),'file':entry})
            evaluate_checkpoint(model, controller, p, q, offsets, label, seen)
    assert seen == c['train_tokens']
    atomic_json(ROOT/'training.json', {'status':'complete','tokens':seen,
        'seconds':elapsed+time.monotonic()-started,'source_fingerprint':fp,
        'gradient_update_windows':update_windows,'config':c,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'backbone_frozen':all(not v.requires_grad and v.grad is None for v in model.parameters())})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['pilot','train'])
    args = parser.parse_args(); ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == 'pilot':
        run(pilot=True)
    else:
        try:
            run(pilot=False)
        except BaseException as error:
            atomic_json(ROOT/'completion.json', {'status':'failed',
                'error_type':type(error).__name__, 'error':str(error)})
            raise
        else:
            atomic_json(ROOT/'completion.json', {'status':'complete'})


if __name__ == '__main__': main()
