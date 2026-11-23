"""Build a fingerprinted dense-trace dataset; fit scaling on train only."""
import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.phase0 import coverage_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('data/stage6-phase0-v1'))
    a = p.parse_args()
    torch.set_num_threads(4)
    corpus = a.root / 'stage3-corpus-v1'
    collection = a.root / 'stage3-full-v1'
    clusters = a.root / 'stage4-clusters-v1'
    signals = a.root / 'stage5-signals-v1'
    cfg = json.loads(Path('configs/phase0.json').read_text())
    fp = {name: file_sha256(folder / 'manifest.json') for name, folder in
          [('corpus', corpus), ('collection', collection), ('clusters', clusters), ('signals', signals)]}
    fp.update({name: file_sha256(Path(name)) for name in
               ['scripts/prepare_phase0.py', 'pathpred/phase0.py', 'configs/phase0.json']})
    out_manifest = a.output / 'manifest.json'
    if out_manifest.exists():
        old = json.loads(out_manifest.read_text())
        if old['fingerprint'] != fp:
            raise ValueError('Existing dataset fingerprint differs')
        if old['status'] == 'complete':
            for entry in old['files'].values():
                assert file_sha256(a.output / entry['file']) == entry['sha256']
            print('Dataset already complete and verified', flush=True)
            return
    atomic_json(out_manifest, {'status': 'in_progress', 'fingerprint': fp})
    sm = json.loads((signals / 'manifest.json').read_text())
    cm = json.loads((clusters / 'manifest.json').read_text())
    coll = json.loads((collection / 'manifest.json').read_text())
    assert sm['status'] == cm['status'] == coll['status'] == 'complete'
    assert sm['fingerprint']['collection_sha256'] == fp['collection']
    assert sm['fingerprint']['clustering_sha256'] == fp['clusters']
    assert sm['fingerprint']['corpus_sha256'] == fp['corpus']
    entry = sm['files']['observations']
    assert file_sha256(signals / entry['file']) == entry['sha256']
    obs = load_file(str(signals / entry['file']))
    n, layers, units = obs['binary'].shape
    hidden = torch.empty(n, layers, 2048, dtype=torch.bfloat16)
    heads = torch.empty(n, layers, 16)
    offset = 0
    for i, shard in enumerate(coll['shards']):
        path = collection / shard['file']
        assert file_sha256(path) == shard['sha256']
        with safe_open(str(path), framework='pt') as f:
            h = f.get_tensor('sample_head_norms')
            k = len(h)
            assert torch.equal(f.get_tensor('sample_sequence_ids'), obs['sequence_ids'][offset:offset+k])
            assert torch.equal(f.get_tensor('sample_token_positions'), obs['token_positions'][offset:offset+k])
            heads[offset:offset+k] = h
            hidden[offset:offset+k] = f.get_tensor('sample_residuals')[:, 1:]
            offset += k
        if i % 10 == 0:
            print(json.dumps({'verified_shards': i+1}), flush=True)
    assert offset == n
    groups = []
    for layer in cm['layers']:
        e = layer['tensors']
        assert file_sha256(clusters / e['file']) == e['sha256']
        groups.append(load_file(str(clusters / e['file']))['sample_group_norms'])
    norms = torch.cat((heads, torch.stack(groups, 1)), -1)
    labels, shares = coverage_labels(norms, cfg['target_norm_coverage'], cfg['target_families'])
    assert torch.isfinite(hidden).all()
    train = obs['split_codes'] == 0
    # Chunk accumulation avoids a second multi-GB temporary and uses train only.
    sumsq = torch.zeros(layers, 2048, dtype=torch.float64)
    ids = train.nonzero().flatten()
    for batch in ids.split(512):
        sumsq += hidden[batch].double().square().sum(0)
    rms = (sumsq / len(ids)).sqrt().float()
    rms[rms == 0] = 1
    rows = [json.loads(x) for x in (corpus / 'sequences.jsonl').read_text().splitlines()]
    docnames = sorted({r['document_sha256'] for r in rows})
    domains = sorted({r['domain'] for r in rows})
    docmap = {s: i for i, s in enumerate(docnames)}
    domainmap = {s: i for i, s in enumerate(domains)}
    documents = torch.tensor([docmap[rows[s]['document_sha256']] for s in obs['sequence_ids'].tolist()])
    domain_ids = torch.tensor([domainmap[rows[s]['domain']] for s in obs['sequence_ids'].tolist()])
    for doc in documents.unique():
        assert obs['split_codes'][documents == doc].unique().numel() == 1
    paired = obs['previous_sample_indices'] >= 0
    for code, expected in [(0, 8176), (1, 900), (2, 924)]:
        assert int((paired & (obs['split_codes'] == code)).sum()) == expected
    files = {'examples': atomic_tensors(a.output / 'examples.safetensors',
        {**obs, 'labels': labels, 'shares': shares, 'documents': documents, 'domains': domain_ids}),
        'hidden': atomic_tensors(a.output / 'hidden.safetensors', {'hidden': hidden, 'rms': rms})}
    atomic_json(out_manifest, {'status': 'complete', 'fingerprint': fp, 'files': files,
        'positions': n, 'domains': domains, 'documents': docnames,
        'scaling': 'per-coordinate RMS fitted on train positions only; divide at load',
        'source_hidden_index': 'completed layer l uses sample_residuals[l+1]',
        'targets': '95% norm prefixes separately for heads and MLP groups; proxy only'})
    print('Phase 0 dataset complete; test metrics not evaluated', flush=True)


if __name__ == '__main__':
    main()
