"""Train-only per-unit quartiles and aligned binary/coarse/magnitude observations.

This creates INPUT observations, not importance targets, and trains no predictor.
Raw values for skipped units are hidden later by the same censoring function used
for current and previous tokens in Phase 0.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.path_signals import UnitThresholds, preceding_sample_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--clusters', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    fingerprint = {'corpus_sha256': file_sha256(args.corpus / 'manifest.json'),
                   'collection_sha256': file_sha256(args.collection / 'manifest.json'),
                   'clustering_sha256': file_sha256(args.clusters / 'manifest.json'),
                   'signals_source_sha256': file_sha256(Path('pathpred/path_signals.py')),
                   'runner_sha256': file_sha256(Path(__file__)), 'upper_quantile': .75,
                   'lower_quantile': .25, 'interpolation': 'linear', 'fit_split': 'train'}
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not args.resume or old['fingerprint'] != fingerprint:
            raise ValueError('Existing output requires matching --resume inputs/code')
        if old['status'] == 'complete':
            for entry in old['files'].values():
                assert file_sha256(args.output / entry['file']) == entry['sha256']
            print('Signal preparation already complete and verified')
            return
    atomic_json(manifest_path, {'status': 'in_progress', 'fingerprint': fingerprint})
    cm = json.loads((args.clusters / 'manifest.json').read_text())
    collection = json.loads((args.collection / 'manifest.json').read_text())
    corpus = json.loads((args.corpus / 'manifest.json').read_text())
    assert cm['status'] == collection['status'] == 'complete'
    assert cm['fingerprint']['collection_manifest_sha256'] == fingerprint['collection_sha256']
    assert cm['fingerprint']['corpus_manifest_sha256'] == fingerprint['corpus_sha256']
    for entry in cm['outputs'].values():
        assert file_sha256(args.clusters / entry['file']) == entry['sha256']
    index = load_file(str(args.clusters / cm['outputs']['sample_index']['file']))
    rows_meta = next(e for e in corpus['files'] if e['file'] == 'sequences.jsonl')
    assert file_sha256(args.corpus / 'sequences.jsonl') == rows_meta['sha256']
    rows = [json.loads(x) for x in (args.corpus / 'sequences.jsonl').read_text().splitlines()]
    split_codes = {'train': 0, 'validation': 1, 'test': 2}
    assert index['split_codes'].tolist() == [split_codes[rows[s]['split']] for s in index['sequence_ids'].tolist()]
    heads, offset = [], 0
    for i, shard in enumerate(collection['shards']):
        path = args.collection / shard['file']
        assert file_sha256(path) == shard['sha256']
        with safe_open(str(path), framework='pt') as f:
            head = f.get_tensor('sample_head_norms')
            n = len(head)
            assert torch.equal(f.get_tensor('sample_sequence_ids'), index['sequence_ids'][offset:offset + n])
            assert torch.equal(f.get_tensor('sample_token_positions'), index['token_positions'][offset:offset + n])
            heads.append(head.clone()); offset += n
        if i % 10 == 0:
            print(json.dumps({'input_shards_checked': i + 1}), flush=True)
    head_norms = torch.cat(heads)
    groups = []
    for layer in cm['layers']:
        entry = layer['tensors']
        assert file_sha256(args.clusters / entry['file']) == entry['sha256']
        groups.append(load_file(str(args.clusters / entry['file']))['sample_group_norms'])
    norms = torch.cat((head_norms, torch.stack(groups, 1)), dim=-1)
    arch = collection['model']['expected_architecture']
    groups_per_layer = arch['intermediate_size'] // cm['fingerprint']['config']['group_size']
    assert norms.shape == (collection['target_samples'], arch['num_hidden_layers'], arch['num_attention_heads'] + groups_per_layer)
    assert torch.isfinite(norms).all() and norms.ge(0).all()
    train = index['split_codes'] == 0
    thresholds = UnitThresholds.fit(norms, train)
    observed = torch.ones_like(norms, dtype=torch.bool)
    binary = thresholds.encode(norms, observed, 'binary')
    coarse = thresholds.encode(norms, observed, 'coarse')
    magnitudes = thresholds.encode(norms, observed, 'magnitudes')
    previous_indices = preceding_sample_indices(index['sequence_ids'], index['token_positions'])
    pair_targets = previous_indices >= 0
    assert int(pair_targets.sum()) == corpus['sample_pairs']
    assert torch.equal(index['split_codes'][previous_indices[pair_targets]], index['split_codes'][pair_targets])
    # Independent, per-unit sorted-sample calculation verifies every upper threshold.
    ordered = norms[train].sort(dim=0).values
    at = .75 * (len(ordered) - 1)
    low, high = int(at), min(int(at) + 1, len(ordered) - 1)
    expected = ordered[low] + (ordered[high] - ordered[low]) * (at - low)
    torch.testing.assert_close(thresholds.upper, expected, rtol=1e-6, atol=1e-7)
    assert torch.equal(binary, (norms > thresholds.upper).to(torch.int8))
    files = {
        'thresholds': atomic_tensors(args.output / 'thresholds.safetensors', {'q75': thresholds.upper, 'q25': thresholds.lower}),
        'observations': atomic_tensors(args.output / 'observations.safetensors',
            {'binary': binary, 'coarse': coarse, 'magnitudes': magnitudes,
             **index, 'previous_sample_indices': previous_indices}),
    }
    frequencies = {}
    for name, code in split_codes.items():
        selected = index['split_codes'] == code
        per_unit = binary[selected].float().mean(0)
        frequencies[name] = {'positions': int(selected.sum()),
                            'paired_targets': int((selected & pair_targets).sum()),
                            'strong_fraction_mean': float(per_unit.mean()),
                            'strong_fraction_min': float(per_unit.min()),
                            'strong_fraction_max': float(per_unit.max())}
    manifest = {'status': 'complete', 'fingerprint': fingerprint, 'files': files,
                'created_at': datetime.now(timezone.utc).isoformat(), 'shape': list(binary.shape),
                'unit_order': f"{arch['num_attention_heads']} query heads then {groups_per_layer} learned MLP groups in each layer",
                'input_norm_definition': 'L2 norm before attention output projection or MLP down projection, before gate scaling',
                'states': {'unobserved': -1, 'observed_not_strong': 0, 'observed_strong': 1},
                'coarse_states': {'unobserved': -1, 'at_or_below_q25': 0, 'above_q25_through_q75': 1, 'above_q75': 2},
                'split_frequencies': frequencies, 'fit_positions': int(train.sum()),
                'checks': ['all source hashes and sample alignment', 'training-only quartiles',
                           'all q75 thresholds independently verified', 'strict greater-than bits verified',
                           'previous-token pairs adjacent within same window and split'],
                'notes': ['These are observations, not causal importance or skip targets',
                          'Dense observations are censored before model input in Phase 0; missing is not zero',
                          'Previous-token censoring uses its actual mask, never its hidden dense observations',
                          'Raw strengths do not enter the binary predictor; continuous strengths are ablation-only',
                          'Byte storage is not bit packing; one activity bit is conditional on a known observation mask']}
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
