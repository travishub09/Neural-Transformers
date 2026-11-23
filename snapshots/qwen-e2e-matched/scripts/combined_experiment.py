"""Matched extra training for combined, path-only, and hidden-only predictors.

Requires an explicit protocol file; never chooses the unresolved token budget or
gate recipe at launch. Each (signal,budget) branch starts from its own first-seed
10M end-to-end checkpoint. Existing artifacts are read-only.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, file_sha256
from pathpred.combined_prediction import make_predictor
from pathpred.end_to_end import DualBudget, quality_route
from scripts import e2e_pilot, e2e_full
from scripts.neuron_guessing import backbone, data, CAL
from scripts.calibrate_native import selections, calibrate, evaluate, ids_for

ROOT = Path('artifacts/combined-v1')
SIGNALS = ('combined', 'binary', 'hidden')


def protocol(path):
    c = json.loads(Path(path).read_text())
    if c.get('gate_recipe') != 'hard-threshold-ste':
        raise ValueError('This driver preserves the actual hard-threshold STE recipe; a changed recipe needs a separate protocol')
    if c.get('train_tokens') not in (5_000_000, 10_000_000):
        raise ValueError('Specify 5M or 10M tokens PER budget branch')
    expected = [n for n in (2_000_000, 5_000_000, 10_000_000) if n <= c['train_tokens']]
    if c.get('milestones') != expected:
        raise ValueError('Milestones must include every requested checkpoint up to the final token count')
    if c.get('sequence_length') != 512 or c.get('dense_prefix_layers') != 2 or c.get('attention') != 'dense':
        raise ValueError('Backbone routing protocol changed')
    if not 5 <= c.get('hidden_lr_multiplier', 0) <= 10:
        raise ValueError('New hidden projection needs the declared 5–10x learning rate')
    return c


def choose_windows(records, forbidden, milestones, seed):
    order = [i for i, r in enumerate(records) if r['split'] == 'train' and r['length'] >= 512
             and r['document_sha256'] not in forbidden]
    order.sort(key=lambda i: hashlib.sha256((str(seed) + records[i]['document_sha256'] + ':' +
               str(records[i]['document_token_offset'])).encode()).hexdigest())
    boundaries = [0] + milestones
    required = sum(math.ceil((b-a)/512) for a, b in zip(boundaries, boundaries[1:]))
    if len(order) < required:
        raise ValueError(f'Only {len(order)} fresh windows remain; need {required}. Prepare more training data; do not reuse held-out documents.')
    return order[:required]


def prepare_training(c):
    corpus, records = data(c['training_corpus'])
    forbidden = set()
    for name in c['exclude_corpora']:
        _, old = data(name)
        forbidden.update(r['document_sha256'] for r in old)
    order = choose_windows(records, forbidden, c['milestones'], c['data_seed'])
    audit = {'window_indices': order, 'full_windows': len(order),
        'unique_documents': len({records[i]['document_sha256'] for i in order}),
        'excluded_documents': len(forbidden), 'data_seed': c['data_seed'],
        'tokens_per_branch': c['train_tokens'], 'total_presentations': 6*c['train_tokens'],
        'fresh_relative_to_all_prior_selected_training_and_stage3_documents': True,
        'same_windows_and_order_all_three_predictors_and_both_budgets': True}
    destination = ROOT/'data-audit.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == audit
    else:
        atomic_json(destination, audit)
    return corpus['input_ids'], records, order


def warm_root(signal, budget):
    name = 'e2e-hidden-full-v1' if signal == 'hidden' else 'e2e-full-v1'
    return Path('artifacts')/name/f'budget-{budget}'


def load_warm(signal, budget, device='cuda'):
    root = warm_root(signal, budget)
    checkpoint = root/'predictor-10000000.safetensors'
    meta = json.loads((root/'checkpoint-10000000.json').read_text())
    assert file_sha256(checkpoint) == meta['file']['sha256']
    p = make_predictor(signal).to(device).eval()
    state = load_file(str(checkpoint), device=device)
    if signal == 'combined':
        p.load_path_state(state)
    else:
        p.load_state_dict(state)
    thresholds = json.loads((root/'threshold-10000000.json').read_text())
    offsets = torch.tensor(thresholds['offsets'], device=device)
    return p, offsets, meta


def optimizer_for(p, c):
    groups = [{'params': [v for name, v in p.named_parameters() if name != 'hidden_input_weight'],
               'lr_multiplier': 1.}]
    if p.config.signal == 'combined':
        groups.append({'params': [p.hidden_input_weight], 'lr_multiplier': c['hidden_lr_multiplier']})
    return torch.optim.AdamW(groups, lr=c['learning_rate'], weight_decay=c['weight_decay'])


def fingerprint(config_path, signal, budget):
    c = protocol(config_path)
    files = list(Path('pathpred').glob('*.py')) + [Path(config_path),
        Path('scripts/combined_experiment.py'), Path('scripts/e2e_pilot.py'),
        Path('scripts/e2e_full.py'), Path('scripts/neuron_guessing.py'),
        Path('scripts/calibrate_native.py'), Path('scripts/oracle_quality.py'),
        Path(c['training_corpus'])/'manifest.json', Path(c['corpus_config']),
        Path('scripts/prepare_corpus.py'), Path('data/stage3-corpus-v1/manifest.json'),
        ROOT/'data-audit.json', CAL]
    files += [warm_root(signal, budget)/name for name in
              ('checkpoint-10000000.json', 'threshold-10000000.json')]
    return {str(path): file_sha256(path) for path in sorted(set(files))}


def reliance_checks(model, controller, p, q, root, tokens, corpus, validation, test, budget):
    base = torch.tensor(json.loads((root/f'threshold-{tokens:08d}.json').read_text())['offsets'], device='cuda')
    for mode in ('no_hidden', 'zero_bits', 'no_path'):
        output = root/f'reliance-{mode}'
        output.mkdir(exist_ok=True)
        with p.reliance(mode):
            # Fixed policy measures raw dependence, including changed compute use.
            if not (output/'fixed.json').exists():
                evaluate(model, controller, p, q, base, corpus, test, output/'fixed.json')
            threshold = output/'threshold.json'
            try:
                if threshold.exists():
                    offsets = torch.tensor(json.loads(threshold.read_text())['offsets'], device='cuda')
                else:
                    offsets, _ = calibrate(model, controller, p, q, base, corpus, validation, budget/100, threshold)
            except RuntimeError as error:
                # Failed budget matching is an explicit ablation result, not a
                # reason to discard completed matched primary runs.
                atomic_json(output/'calibration-failure.json', {'error': str(error), 'mode': mode})
                continue
            if not (output/'matched.json').exists():
                evaluate(model, controller, p, q, offsets, corpus, test, output/'matched.json')


def smoke(c, model, controller, q, corpus, records, order, eval_corpus, validation):
    results = {}
    ids = ids_for(eval_corpus, validation[0])[:, :32]
    # Check actual Qwen closed-loop decisions in BF16 at both warm-start budgets.
    for budget in (50, 40):
        snapshots = []
        for signal in ('binary', 'combined'):
            p, offsets, _ = load_warm(signal, budget)
            with torch.no_grad(), quality_route(model, controller, p, q, mode='hard',
                    score_offsets=offsets, checkpoint_layers=False) as trace:
                logits = model(ids, use_cache=False).logits
            snapshots.append((logits.cpu(), {l: g.detach().cpu() for l, g in trace['gates'].items()}))
            del p
        assert torch.equal(snapshots[0][0], snapshots[1][0])
        assert all(torch.equal(snapshots[0][1][l], snapshots[1][1][l]) for l in range(28))
        results[f'warm_start_{budget}'] = {'exact_qwen_logits_and_gates_equal': True, 'tokens': ids.numel()}
    for signal in SIGNALS:
        torch.manual_seed(c['seed'])
        p, offsets, meta = load_warm(signal, 50)
        optimizer = optimizer_for(p, c)
        dual = DualBudget(.5, meta['dual']['alpha'], c['dual_learning_rate'], c['maximum_alpha'])
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for index in range(8):
            row = records[order[index]]
            ids = corpus[row['sequence_id']:row['sequence_id']+1, :512].long().cuda()
            torch.cuda.synchronize(); start = time.monotonic()
            metrics = e2e_pilot.train_batch(model, controller, p, optimizer, q, offsets, ids,
                dual, .3, 'hard', c['learning_rate'])
            torch.cuda.synchronize(); durations.append(time.monotonic()-start)
        results[signal] = {'steps': 8, 'tokens_per_second_after_first_step': 7*512/sum(durations[1:]),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'metrics': metrics,
            'accounting': p.accounting(), 'backbone_frozen': all(not v.requires_grad and v.grad is None for v in model.parameters()),
            'new_projection_changed': bool(p.hidden_input_weight.detach().abs().sum()) if signal == 'combined' else None}
        if signal == 'combined': assert results[signal]['new_projection_changed']
        del optimizer, p
        torch.cuda.empty_cache()
    atomic_json(ROOT/'gpu-smoke.json', results)
    print(json.dumps(results), flush=True)


def train_branch(c, config_path, signal, budget, model, controller, q, corpus, records,
                 order, eval_corpus, validation, test):
    root = ROOT/signal/f'budget-{budget}'
    root.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(config_path, signal, budget)
    torch.manual_seed(c['seed'])
    p, offsets, meta = load_warm(signal, budget)
    optimizer = optimizer_for(p, c)
    dual = DualBudget(budget/100, meta['dual']['alpha'], c['dual_learning_rate'], c['maximum_alpha'])
    seen = position = last_save = 0
    elapsed = 0.
    if (root/'last.pt').exists():
        state = torch.load(root/'last.pt', map_location='cuda', weights_only=True)
        assert state['fingerprint'] == fp
        p.load_state_dict(state['predictor']); optimizer.load_state_dict(state['optimizer'])
        offsets = state['offsets']; dual = DualBudget(**state['dual'])
        seen = state['tokens']; position = state['position']; elapsed = state['elapsed']; last_save = seen
        torch.set_rng_state(state['cpu_rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        del state
    if seen in c['milestones']:
        e2e_full.checkpoint(model, controller, p, q, root, seen, offsets, dual, fp, eval_corpus, validation, test)
    start = time.monotonic(); step_time = 0.; new_tokens = 0
    torch.cuda.reset_peak_memory_stats()
    for index in range(position, len(order)):
        if seen >= c['train_tokens']: break
        length = min(512, min(n for n in c['milestones'] if n > seen)-seen)
        row = records[order[index]]
        ids = corpus[row['sequence_id']:row['sequence_id']+1, :length].long().cuda()
        factor = min(1., (seen+length)/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/c['train_tokens'])))
        temperature = .3*(.1/.3)**min(1., seen/2_000_000)
        torch.cuda.synchronize(); tick = time.monotonic()
        metrics = e2e_pilot.train_batch(model, controller, p, optimizer, q, offsets, ids, dual,
            temperature, 'hard', c['learning_rate']*factor)
        torch.cuda.synchronize(); step_time += time.monotonic()-tick
        seen += length; new_tokens += length; position = index+1
        if index % 20 == 0 or seen in c['milestones']:
            progress = {'signal': signal, 'budget': budget, 'tokens': seen, 'position': position,
                'mode': 'hard', 'temperature': temperature, 'tokens_per_second': new_tokens/step_time,
                'elapsed_seconds': elapsed+time.monotonic()-start, 'metrics': metrics,
                'learning_rates': [g['lr'] for g in optimizer.param_groups]}
            atomic_json(root/'progress.json', progress); print(json.dumps(progress), flush=True)
        guard = elapsed+time.monotonic()-start > c['maximum_branch_hours']*3600
        if seen-last_save >= c['checkpoint_every_tokens'] or seen in c['milestones'] or guard:
            if shutil.disk_usage(ROOT).free < 3_000_000_000:
                raise RuntimeError('Insufficient volume space for atomic resumable checkpoint')
            e2e_full.save_resume(root, p, optimizer, offsets, dual, seen, position,
                                 elapsed+time.monotonic()-start, fp)
            last_save = seen
        if guard: raise TimeoutError('Branch time guard reached; latest state persisted')
        if seen in c['milestones']:
            e2e_full.checkpoint(model, controller, p, q, root, seen, offsets, dual, fp, eval_corpus, validation, test)
    assert seen == c['train_tokens']
    atomic_json(root/'training.json', {'status': 'complete', 'signal': signal, 'target': budget/100,
        'tokens': seen, 'gradient_update_windows': position, 'seconds': elapsed+time.monotonic()-start,
        'fingerprint': fp, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'backbone_frozen': all(not v.requires_grad and v.grad is None for v in model.parameters())})
    if signal == 'combined':
        reliance_checks(model, controller, p, q, root, seen, eval_corpus, validation, test, budget)
    atomic_json(root/'completion.json', {'status': 'complete', 'fingerprint': fp})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--signal', choices=SIGNALS)
    parser.add_argument('--budget', type=int, choices=[50, 40])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.smoke and (args.signal is None or args.budget is None):
        parser.error('Training requires an explicit signal and budget')
    c = protocol(args.config); e2e_pilot.config = lambda: c
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    corpus, records, order = prepare_training(c)
    eval_corpus, validation, test = selections()
    model, controller = backbone(); q = load_file(str(CAL), device='cuda')['q75']
    if args.smoke:
        smoke(c, model, controller, q, corpus, records, order, eval_corpus, validation)
    else:
        train_branch(c, args.config, args.signal, args.budget, model, controller, q,
                     corpus, records, order, eval_corpus, validation, test)


if __name__ == '__main__': main()
