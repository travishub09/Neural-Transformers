"""Freeze validation-selected checkpoints, then run the fixed held-out protocol."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, file_sha256
from pathpred.path_model import PathModel, PathConfig
from pathpred.phase0 import select_units, metric_rows, summarize_rows, paired_document_bootstrap, normalized_skill_bootstrap
from pathpred.phase0_training import PhaseData, evaluate_model, run_name, training_fingerprint


def baseline_scores(data):
    ids = data.ids['train']
    return {'frequency': data.t['labels'][ids, 2:].float().mean(0),
            'mean_norm': data.t['shares'][ids, 2:].mean(0)}


@torch.no_grad()
def static_rows(data, split, scores, counts, cfg, seed=0):
    parts = []
    gen = torch.Generator(device=data.device).manual_seed(seed)
    for ids in data.ids[split].split(cfg['batch_size']):
        labels = data.t['labels'][ids, 2:]
        shares = data.t['shares'][ids, 2:]
        if isinstance(scores, str):
            if scores == 'random':
                values = torch.rand(labels.shape, device=data.device, generator=gen)
            elif scores == 'oracle':
                values = labels.float() + shares * .1
            else:
                raise ValueError(scores)
        else:
            values = scores.unsqueeze(0).expand(len(ids), -1, -1)
        rows = metric_rows(select_units(values, counts), labels, shares, data.t['binary'][ids, 2:] == 1)
        parts.append({k: v.cpu().numpy() for k, v in rows.items()})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def mean_rows(rows):
    return {k: np.stack([r[k] for r in rows]).mean(0) for k in rows[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/stage6-phase0-v1'))
    p.add_argument('--output', type=Path, default=Path('artifacts/phase0-v1'))
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--config', type=Path, default=Path('configs/phase0.json'))
    p.add_argument('--quality-report', type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    training = json.loads((a.output / 'training.json').read_text())
    assert training['status'] == 'complete' and not training['pilot']
    cfg = json.loads(a.config.read_text())
    assert training['fingerprint'] == training_fingerprint(a.data, cfg, a.config)
    quality = None
    if cfg.get('target') == 'strong':
        if a.quality_report is None:
            raise ValueError('Strong-target joint gate requires the completed oracle-quality report')
        quality = json.loads(a.quality_report.read_text())
        assert quality['status'] == 'complete' and not quality['pilot']
    freeze_path = a.output / 'freeze.json'
    if a.freeze:
        if freeze_path.exists():
            print('Already frozen; refusing to replace selection')
            return
        data = PhaseData(a.data, target=cfg.get('target', 'coverage'))
        baselines = baseline_scores(data)
        selected, validation = {}, {}
        for b in cfg['gated_budget_anchors']:
            counts = training['plans'][str(b)]['static']['counts']
            results = {name: summarize_rows(static_rows(data, 'validation', scores, counts, cfg))
                       for name, scores in baselines.items()}
            selected[str(b)] = max(results, key=lambda k: results[k]['recall'])
            validation[str(b)] = results
        checkpoints = {}
        for b in cfg['gated_budget_anchors']:
            for seed in cfg['seeds']:
                for signal in cfg['signals']:
                    name = run_name(signal, b, seed)
                    report = json.loads((a.output / name / 'report.json').read_text())
                    assert report['fingerprint'] == training['fingerprint']
                    sha = file_sha256(a.output / name / 'best.safetensors')
                    assert sha == report['checkpoint_sha256']
                    checkpoints[name] = sha
        atomic_json(freeze_path, {'checkpoint_hashes': checkpoints, 'static_selected_on_validation': selected,
            'static_validation': validation, 'training_manifest_sha256': file_sha256(a.output / 'training.json'),
            'evaluator_sha256': file_sha256(Path(__file__)), 'test_used': False,
            'quality_report_sha256': file_sha256(a.quality_report) if quality else None})
        print(json.dumps({'frozen_checkpoints': len(checkpoints), 'selected_static': selected, 'test_used': False}), flush=True)
        return
    if not freeze_path.exists():
        raise ValueError('Freeze all selections before test evaluation')
    frozen = json.loads(freeze_path.read_text())
    assert frozen['training_manifest_sha256'] == file_sha256(a.output / 'training.json')
    assert frozen['evaluator_sha256'] == file_sha256(Path(__file__))
    if quality:
        assert frozen['quality_report_sha256'] == file_sha256(a.quality_report)
    result_path = a.output / 'results.json'
    if result_path.exists():
        print('Final test results already exist; refusing to overwrite')
        return
    data = PhaseData(a.data, include_test=True, target=cfg.get('target', 'coverage'))
    baselines = baseline_scores(data)
    test_ids = data.ids['test']
    docs = data.t['documents'][test_ids].cpu().numpy()
    domains = data.t['domains'][test_ids].cpu().numpy()
    rowsdir = a.output / 'test-rows'
    rowsdir.mkdir(exist_ok=True)
    results = {}
    for b in cfg['gated_budget_anchors']:
        plans = training['plans'][str(b)]
        all_rows, summary = {}, {}
        masks = [data.mask(plans['binary']['counts'], seed) for seed in cfg['test_mask_seeds']]
        for signal in cfg['signals']:
            seed_rows = []
            for seed in cfg['seeds']:
                name = run_name(signal, b, seed)
                folder = a.output / name
                assert file_sha256(folder / 'best.safetensors') == frozen['checkpoint_hashes'][name]
                report = json.loads((folder / 'report.json').read_text())
                model = PathModel(PathConfig(**report['model_config'])).to(data.device)
                model.load_state_dict(load_file(str(folder / 'best.safetensors'), device=data.device))
                rows = mean_rows([evaluate_model(model, data, 'test', mask, plans[signal]['counts'], cfg)[1]
                                  for mask in masks])
                seed_rows.append(rows)
                np.savez_compressed(rowsdir / (name+'.npz'), **rows, documents=docs, domains=domains)
                del model
            all_rows[signal] = seed_rows
            metrics = [summarize_rows(r) for r in seed_rows]
            summary[signal] = {'mean': summarize_rows(mean_rows(seed_rows)), 'seeds': metrics,
                'recall_seed_sd': float(np.std([m['recall'] for m in metrics], ddof=1)), 'cost': plans[signal]}
            print(json.dumps({'budget': b, 'signal': signal, 'test_recall': summary[signal]['mean']['recall']}), flush=True)
        for name, scores in {**baselines, 'random': 'random', 'oracle': 'oracle'}.items():
            rows = mean_rows([static_rows(data, 'test', scores, plans['static']['counts'], cfg, seed)
                              for seed in cfg['test_mask_seeds']])
            all_rows[name] = [rows] * len(cfg['seeds'])
            summary[name] = {'mean': summarize_rows(rows), 'cost': plans['static'],
                             'uses_ground_truth': name == 'oracle'}
            np.savez_compressed(rowsdir / f'{name}-b{round(b*100):02d}.npz', **rows, documents=docs, domains=domains)
        chosen = frozen['static_selected_on_validation'][str(b)]
        binary = np.stack([r['true_positive'] for r in all_rows['binary']])
        positives = all_rows['binary'][0]['positive']
        comparisons = {}
        for name in ['hidden', chosen, 'coarse', 'magnitudes', 'decisions']:
            other = np.stack([r['true_positive'] for r in all_rows[name]])
            comparisons['binary_minus_'+name] = paired_document_bootstrap(binary, other, positives, docs, domains,
                repetitions=cfg['bootstrap_repetitions'], seed=cfg['bootstrap_seed'])
        domain_results = {}
        for domain_id, domain in enumerate(data.manifest['domains']):
            pick = domains == domain_id
            if not pick.any():
                continue
            domain_results[domain] = {'examples': int(pick.sum()), 'documents': int(len(np.unique(docs[pick]))),
                'methods': {name: summarize_rows({k: v[pick] for k, v in mean_rows(rows).items()})
                            for name, rows in all_rows.items()}}
        results[str(b)] = {'methods': summary, 'comparisons': comparisons,
                           'selected_static': chosen, 'domains': domain_results}
        if cfg.get('target') == 'strong':
            positives_by_layer = data.t['labels'][test_ids, 2:].sum(-1).cpu().numpy()
            total_positives = positives_by_layer.sum()
            normalizations = {}
            for name, rows in all_rows.items():
                counts = np.array(plans[name if name in cfg['signals'] else 'static']['counts'])
                expected = (positives_by_layer * counts[None,:] / data.units).sum(1)
                ceiling = np.minimum(positives_by_layer, counts[None,:]).sum(1)
                hits = np.stack([r['true_positive'] for r in rows])
                normalized = normalized_skill_bootstrap(hits, expected, ceiling, docs, domains,
                    repetitions=cfg['bootstrap_repetitions'], seed=cfg['bootstrap_seed'])
                net_advantage = normalized_skill_bootstrap(hits, all_rows[chosen][0]['true_positive'],
                    all_rows['oracle'][0]['true_positive'], docs, domains,
                    repetitions=cfg['bootstrap_repetitions'], seed=cfg['bootstrap_seed'])
                normalizations[name] = {'normalized_skill': normalized,
                    'expected_random_recall_at_method_counts':float(expected.sum()/total_positives),
                    'perfect_ranker_recall_at_method_counts':float(ceiling.sum()/total_positives),
                    'fraction_of_oracle_fixed_advantage_at_common_cap':net_advantage}
            results[str(b)]['normalizations'] = normalizations
        atomic_json(a.output / 'evaluation-progress.json', results)
        del masks
    primary = results[str(cfg['primary_budget'])]
    static_diff = primary['comparisons']['binary_minus_'+primary['selected_static']]
    hidden_diff = primary['comparisons']['binary_minus_hidden']
    static_pass = static_diff['difference'] >= cfg['minimum_static_gain'] and static_diff['ci95'][0] > 0
    hidden_pass = hidden_diff['ci95'][0] > -cfg['noninferiority_margin']
    report = {'status': 'complete', 'freeze_sha256': file_sha256(freeze_path), 'results': results,
        'decision': {'provisional_go': bool(static_pass and hidden_pass), 'static_gain_pass': bool(static_pass),
                     'hidden_noninferiority_pass': bool(hidden_pass), 'primary_budget': cfg['primary_budget'],
                     'meaning': 'Proxy predictability only; no sparse LLM quality or speed established'},
        'test_examples': len(test_ids), 'test_documents': len(np.unique(docs)),
        'test_masks': cfg['test_mask_seeds'], 'seeds': cfg['seeds'],
        'limitations': ['Dense activation replay with exogenous masks, not a sparse student',
            'Norm coverage labels are proxies, not causal importance',
            'Three seeds and limited document sample; no hyperparameter search',
            'Theoretical matmul budgets exclude kernel/launch overhead',
            'Hidden comparator has more parameters and scalar information than binary']}
    if cfg.get('target') == 'strong':
        gain = primary['normalizations']['binary']['fraction_of_oracle_fixed_advantage_at_common_cap']
        prediction_pass = gain['value'] is not None and gain['value'] >= cfg['minimum_oracle_fixed_gap_recovered'] and static_diff['ci95'][0] > 0
        oracle_key = 'oracle_binary-b'+str(round(cfg['primary_budget']*100))
        quality_pass = quality['results'][oracle_key]['quality_pass_provisional']
        report['decision'] = {'provisional_go':bool(prediction_pass and quality_pass),
            'prediction_gain_pass':bool(prediction_pass),'binary_activity_oracle_quality_pass':bool(quality_pass),
            'minimum_oracle_fixed_gap_recovered':cfg['minimum_oracle_fixed_gap_recovered'],
            'observed_oracle_fixed_gap_recovered':gain['value'], 'primary_budget':cfg['primary_budget'],
            'quality_report_sha256':file_sha256(a.quality_report),
            'quality_threshold_status':quality['config']['quality_threshold_status'],
            'meaning':'Joint activity-oracle quality and prediction gate; no real speedup established'}
        report['target'] = 'above own training-only q75, strict greater-than'
        report['limitations'] = [
            'Strong/weak activity labels are not causal importance or output-quality targets'
            if text == 'Norm coverage labels are proxies, not causal importance' else text
            for text in report['limitations']]
        report['limitations'].append('Same corpus partitions as original experiment; exploratory follow-up, not independent replication')
    # Descriptive target diagnostics, computed only after fixed final evaluation.
    target_diagnostics = {}
    for split in ['train', 'validation', 'test']:
        labels = data.t['labels'][data.ids[split], 2:].float()
        p = labels.mean(0)
        target_diagnostics[split] = {
            'examples': len(labels), 'positive_fraction': float(labels.mean()),
            'head_positive_fraction': float(labels[..., :16].mean()),
            'mlp_positive_fraction': float(labels[..., 16:].mean()),
            'unit_layer_targets_positive_at_least_99_percent': int((p >= .99).sum()),
            'unit_layer_targets_total': p.numel()}
    report['target_diagnostics'] = target_diagnostics
    atomic_json(result_path, report)
    print(json.dumps(report['decision']), flush=True)


if __name__ == '__main__':
    main()
