"""Six-document screen of magnitude-based MLP neuron gating; attention dense."""
import json
import math
from pathlib import Path
import time

import torch
from safetensors.torch import load_file
from pathpred import gate_qwen3
from pathpred.collection import atomic_json, file_sha256
from pathpred.reordering import load_clustered_qwen
from pathpred.oracle_checks import quality_rows
from pathpred.sparsity_diagnostics import Condition, DynamicPolicy
from scripts.oracle_quality import select_documents, aggregate


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads(Path('configs/neuron-quick.json').read_text())
    root = Path('data/stage3-corpus-v1')
    output = Path('artifacts/neuron-quick-v1')
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'results.json').exists(), 'Preserve prior completed results'
    corpus = json.loads((root/'manifest.json').read_text())
    for entry in corpus['files']:
        assert file_sha256(root/entry['file']) == entry['sha256']
    records = [json.loads(line) for line in (root/'sequences.jsonl').read_text().splitlines()]
    selected = select_documents(records, 'test', cfg['documents_per_domain'], cfg)
    assert len(selected) == 6
    fp = {s: file_sha256(Path(s)) for s in [
        'scripts/quick_neuron_budgets.py', 'configs/neuron-quick.json',
        'pathpred/gated_qwen.py', 'pathpred/reordering.py',
        'pathpred/sparsity_diagnostics.py', 'pathpred/oracle_checks.py',
        'scripts/oracle_quality.py', 'data/stage3-corpus-v1/manifest.json']}
    manifest = dict(status='in_progress', config=cfg, fingerprint=fp, documents=selected,
                    rule='top-k absolute SwiGLU intermediate activation per token per layer',
                    attention='fully dense', protections='none', group_size=1)
    atomic_json(output/'manifest.json', manifest)
    tokens = load_file(str(root/'tokens.safetensors'))['input_ids']
    model = load_clustered_qwen('checkpoints/qwen3-1.7b-base-clustered-v1',
        dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='eager', local_files_only=True)
    model.eval()
    controller = gate_qwen3(model, 1)
    policy = DynamicPolicy(model)
    conditions = [Condition('mlp', 1, 'magnitude', k) for k in cfg['kept_fractions']]
    documents = []
    with torch.inference_mode():
        for i, meta in enumerate(selected):
            length = min(meta['length'], cfg['maximum_sequence_length'])
            ids = tokens[meta['sequence_id']:meta['sequence_id']+1, :length].long().cuda()
            with controller.use_gates():
                teacher = model(ids, use_cache=False).logits
            if i == 0:
                with controller.use_gates(), policy.apply(Condition('mlp', 1, 'magnitude', 1), controller):
                    assert torch.equal(teacher, model(ids, use_cache=False).logits)
            doc = dict(metadata=meta, conditions={})
            for condition in conditions:
                assert time.monotonic()-started < cfg['maximum_runtime_seconds'], 'Runtime guard'
                with controller.use_gates(), policy.apply(condition, controller):
                    student = model(ids, use_cache=False).logits
                metrics = quality_rows(teacher, student, ids)
                row = {k: float(v.double().sum()) for k, v in metrics.items()}
                retained = policy.fractions()
                assert retained['attention'] == 1
                assert retained['mlp'] == math.ceil(6144*condition.keep)/6144
                row.update(tokens=length-1, domain=meta['domain'], document=meta['document_sha256'],
                           retained=retained, max_token_kl=float(metrics['kl'].max()))
                if condition.keep == .5:
                    old = json.loads(Path(f'artifacts/sparsity-calibration-v2/calibration-window-{meta["sequence_id"]}.json').read_text())
                    for key in ['teacher_nll', 'student_nll', 'kl', 'top1_agreement']:
                        assert row[key] == old['conditions'][condition.name][key], (meta['sequence_id'], key)
                doc['conditions'][condition.name] = row
                del student
            atomic_json(output/f'window-{meta["sequence_id"]}.json', doc)
            documents.append(doc)
            del teacher
            print(json.dumps({'documents_complete': i+1, 'seconds': time.monotonic()-started}), flush=True)
    results = {}
    for condition in conditions:
        rows = [d['conditions'][condition.name] for d in documents]
        results[condition.name] = dict(overall=aggregate(rows),
            domains={d: aggregate([r for r in rows if r['domain'] == d]) for d in ['web','code','conversation']},
            retained=rows[0]['retained'])
    report = dict(status='complete', config=cfg, fingerprint=fp, results=results,
                  documents=len(documents), seconds=time.monotonic()-started,
                  checks=dict(all_on_bitwise_equal=True, old_50_percent_metrics_exact=True,
                              attention_dense=True, exact_neuron_counts=True),
                  limitation='Six-document exploratory screen; dense masking, no learned predictor or speed claim.')
    atomic_json(output/'results.json', report)
    manifest['status'] = 'complete'
    atomic_json(output/'manifest.json', manifest)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
