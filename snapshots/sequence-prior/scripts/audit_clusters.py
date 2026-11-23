"""Independently check permutations and every regrouped sampled norm."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from pathpred.collection import atomic_json, file_sha256
from pathpred.clustering import validate_permutation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clusters', type=Path, required=True)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cm = json.loads((args.clusters / 'manifest.json').read_text())
    assert cm['status'] == 'complete'
    assert file_sha256(args.collection / 'manifest.json') == cm['fingerprint']['collection_manifest_sha256']
    source = json.loads((args.collection / 'manifest.json').read_text())
    for entry in cm['outputs'].values():
        assert file_sha256(args.clusters / entry['file']) == entry['sha256']
    combined = load_file(str(args.clusters / cm['outputs']['permutations']['file']))
    index = load_file(str(args.clusters / cm['outputs']['sample_index']['file']))
    assert file_sha256(args.corpus / 'manifest.json') == cm['fingerprint']['corpus_manifest_sha256']
    corpus_manifest = json.loads((args.corpus / 'manifest.json').read_text())
    record_file = next(x for x in corpus_manifest['files'] if x['file'] == 'sequences.jsonl')
    assert file_sha256(args.corpus / 'sequences.jsonl') == record_file['sha256']
    records = [json.loads(x) for x in (args.corpus / 'sequences.jsonl').read_text().splitlines()]
    codes = {'train': 0, 'validation': 1, 'test': 2}
    expected_splits = torch.tensor([codes[records[s]['split']] for s in index['sequence_ids'].tolist()], dtype=torch.int32)
    assert torch.equal(index['split_codes'], expected_splits)
    for name, code in codes.items():
        assert int(expected_splits.eq(code).sum()) == cm['samples_by_split'][name]
    size = cm['fingerprint']['config']['group_size']
    layer_norms, assignments = [], []
    for layer, entry in enumerate(cm['layers']):
        assert layer == entry['layer']
        for meta in [entry['tensors'], entry['metrics_file']]:
            assert file_sha256(args.clusters / meta['file']) == meta['sha256']
        tensors = load_file(str(args.clusters / entry['tensors']['file']))
        perm = tensors['permutation']
        validate_permutation(perm, source['model']['expected_architecture']['intermediate_size'])
        assert torch.equal(perm, combined['permutations'][layer])
        assert torch.equal(tensors['inverse_permutation'], torch.argsort(perm))
        assert torch.equal(combined['inverse_permutations'][layer], torch.argsort(perm))
        groups = torch.empty_like(perm)
        groups[perm] = torch.arange(len(perm) // size).repeat_interleave(size)
        assert torch.bincount(groups).eq(size).all()
        norms = tensors['sample_group_norms']
        assert norms.shape == (source['target_samples'], len(perm) // size)
        assert torch.isfinite(norms).all() and norms.ge(0).all()
        assignments.append(groups); layer_norms.append(norms)
    offset = 0
    worst = 0.
    for shard in source['shards']:
        path = args.collection / shard['file']
        assert file_sha256(path) == shard['sha256']
        with safe_open(str(path), framework='pt') as f:
            raw = f.get_tensor('sample_mlp')
            ns = len(raw)
            assert torch.equal(index['sequence_ids'][offset:offset + ns], f.get_tensor('sample_sequence_ids'))
            assert torch.equal(index['token_positions'][offset:offset + ns], f.get_tensor('sample_token_positions'))
            for layer, groups in enumerate(assignments):
                # Independent original-index bucket summation; no permutation/reshape norm code.
                sums = torch.zeros(ns, len(groups) // size)
                sums.scatter_add_(1, groups.expand(ns, -1), raw[:, layer].float().square())
                recomputed = sums.sqrt()
                stored = layer_norms[layer][offset:offset + ns]
                torch.testing.assert_close(stored, recomputed, rtol=2e-6, atol=1e-6)
                if ns:
                    worst = max(worst, float(((stored - recomputed).abs() / recomputed.clamp_min(1e-6)).max()))
            offset += ns
    assert offset == source['target_samples']
    report = {'status': 'passed', 'layers': len(assignments), 'groups_per_layer': len(assignments[0]) // size,
              'neurons_per_group': size, 'samples': offset,
              'norms_checked': sum(t.numel() for t in layer_norms), 'max_relative_norm_error': worst,
              'manifest_sha256': file_sha256(args.clusters / 'manifest.json'),
              'auditor_sha256': file_sha256(Path(__file__)),
              'checks': ['all input and output checksums', 'bijections and exact capacities',
                         'inverse and combined permutation consistency', 'sample alignment and corpus split identities',
                         'every sampled group norm independently recomputed by original neuron index']}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
