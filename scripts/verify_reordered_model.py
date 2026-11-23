"""Verify the learned permutations on pretrained weights, save/reload the model.

Criteria are declared before running: FP32 max logit error <= 2e-4;
BF16 relative RMS logit error <= .01 and mean teacher KL <= .001.
We report numerical equivalence, not bitwise equivalence after permutation.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathpred import gate_qwen3
from pathpred.clustering import reorder_mlp_, unit_costs, validate_permutation
from pathpred.collection import atomic_json, file_sha256
from pathpred.reordering import enable_canonical_mlp_order, load_clustered_qwen


def compare(a, b, mask=None):
    if mask is not None:
        a, b = a[mask.bool()], b[mask.bool()]
    a, b = a.float(), b.float()
    delta = a - b
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    kl = (la.exp() * (la - lb)).sum(-1)
    return {"max_abs": float(delta.abs().max()), "mean_abs": float(delta.abs().mean()),
            "relative_rms": float(delta.square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-12)),
            "mean_kl": float(kl.mean()), "max_token_kl": float(kl.max()),
            "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
            "positions": a.numel() // a.shape[-1], "bitwise_equal": torch.equal(a, b)}


def check(result, dtype):
    if not all(torch.isfinite(torch.tensor(result[k])) for k in ["max_abs", "relative_rms", "mean_kl"]):
        raise AssertionError("Nonfinite output comparison")
    if dtype == "float32":
        assert result["max_abs"] <= 2e-4, result
    else:
        assert result["relative_rms"] <= .01 and result["mean_kl"] <= .001, result


def cast_preserving_rotary(model, dtype):
    # from_pretrained keeps RoPE frequencies in FP32. A blanket model.to(bf16)
    # would quantize nonpersistent frequency buffers and change the function.
    rotary = model.model.rotary_emb
    buffers = {name: value.clone() for name, value in rotary.named_buffers()}
    model.to(dtype)
    for name, value in buffers.items():
        setattr(rotary, name, value)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {"status": "running", "criteria": {"fp32_max_abs": 2e-4, "bf16_relative_rms": .01, "bf16_mean_kl": .001},
              "cluster_manifest_sha256": file_sha256(args.clusters / "manifest.json"),
              "verifier_sha256": file_sha256(Path(__file__)), "checks": {}, "logit_comparisons": {},
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    report['execution_mode'] = 'canonical_original_order_dense_reference'
    report['reordering_sha256'] = file_sha256(Path('pathpred/reordering.py'))
    report['gating_sha256'] = file_sha256(Path('pathpred/gated_qwen.py'))
    report['bf16_reduced_precision_reduction'] = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    cm = json.loads((args.clusters / "manifest.json").read_text())
    assert cm["status"] == "complete"
    assert file_sha256(args.corpus / "manifest.json") == cm["fingerprint"]["corpus_manifest_sha256"]
    perm_meta = cm["outputs"]["permutations"]
    assert file_sha256(args.clusters / perm_meta["file"]) == perm_meta["sha256"]
    permutations = load_file(str(args.clusters / perm_meta["file"]))["permutations"]
    cfg = cm["model"]
    path = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    original = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="eager", local_files_only=True).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = copy.deepcopy(original)
    assert permutations.shape == (model.config.num_hidden_layers, model.config.intermediate_size)
    for block, perm in zip(model.model.layers, permutations):
        validate_permutation(perm, model.config.intermediate_size)
        reorder_mlp_(block.mlp, perm)
    # Verify the exact intended tensor permutation, then its inverse, for ALL weights.
    expected = dict(original.named_parameters())
    for name, value in model.named_parameters():
        target = expected[name]
        if '.mlp.' in name:
            layer = int(name.split('.')[2]); perm = permutations[layer].to('cuda')
            if name.endswith('gate_proj.weight') or name.endswith('up_proj.weight'):
                target = target[perm]
            elif name.endswith('down_proj.weight'):
                target = target[:, perm]
        assert torch.equal(value, target), name
    report["checks"]["all_weight_permutations_exact"] = True
    for block, perm in zip(model.model.layers, permutations):
        reorder_mlp_(block.mlp, torch.argsort(perm))
    assert all(torch.equal(p, expected[n]) for n, p in model.named_parameters())
    report["checks"]["inverse_restores_all_weights_bitwise"] = True
    for block, perm in zip(model.model.layers, permutations):
        reorder_mlp_(block.mlp, perm)
    enable_canonical_mlp_order(model, permutations)
    # Predetermined held-out windows: first two per domain, never selected by outcomes.
    corpus_manifest = json.loads((args.corpus / 'manifest.json').read_text())
    for entry in corpus_manifest['files']:
        assert file_sha256(args.corpus / entry['file']) == entry['sha256']
    records = [json.loads(x) for x in (args.corpus / 'sequences.jsonl').read_text().splitlines()]
    corpus = load_file(str(args.corpus / 'tokens.safetensors'))
    batches = []
    selected = []
    for domain in ('web', 'code', 'conversation'):
        rows = [r for r in records if r['split'] == 'test' and r['domain'] == domain][:2]
        assert len(rows) == 2
        for r in rows:
            seq, length = r['sequence_id'], r['length']
            ids = corpus['input_ids'][seq:seq + 1, :length].long().to('cuda')
            batches.append((f'{domain}-{seq}', {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}))
            selected.append({k: r[k] for k in ['sequence_id', 'domain', 'split', 'length', 'document_sha256']})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    batches.append(('left_padded', tokenizer(['Hello.', 'Please explain why a frozen model can be reorganized without changing its mathematical function.'], return_tensors='pt', padding=True).to('cuda')))
    report['heldout_windows'] = selected
    for dtype in ('bfloat16', 'float32'):
        cast_preserving_rotary(original, getattr(torch, dtype))
        cast_preserving_rotary(model, getattr(torch, dtype))
        results = {}
        for name, inputs in batches:
            with torch.inference_mode():
                a = original(**inputs, use_cache=False).logits
                b = model(**inputs, use_cache=False).logits
                result = compare(a, b, inputs['attention_mask'])
                check(result, dtype)
            results[name] = result
            del a, b
            print(json.dumps({'dtype': dtype, 'case': name, **result}), flush=True)
        # Cached continuation uses identical input tokens in both models.
        ids = batches[0][1]['input_ids'][:, :16]
        cache_a = cache_b = None
        cached = []
        with torch.inference_mode():
            for begin, end in [(0, 8), (8, 9), (9, 12), (12, 16)]:
                a = original(ids[:, begin:end], past_key_values=cache_a, use_cache=True)
                b = model(ids[:, begin:end], past_key_values=cache_b, use_cache=True)
                result = compare(a.logits, b.logits)
                check(result, dtype)
                cached.append(result)
                cache_a, cache_b = a.past_key_values, b.past_key_values
        del a, b, cache_a, cache_b
        report['logit_comparisons'][dtype] = {'windows': results, 'cached_chunks': cached}
        atomic_json(args.output, report)
    cast_preserving_rotary(model, torch.bfloat16)
    del original
    torch.cuda.empty_cache()
    # A compact permutation artifact is authoritative; also save a ready-to-load checkpoint.
    if args.checkpoint.exists():
        raise FileExistsError('Refusing to overwrite a checkpoint')
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    weight_size = sum(p.numel() * p.element_size() for p in model.parameters())
    if shutil.disk_usage(args.checkpoint.parent).free < weight_size + 2 * 1024**3:
        raise RuntimeError('Insufficient free space for reordered checkpoint and reserve')
    temp = args.checkpoint.with_name(args.checkpoint.name + '.partial')
    if temp.exists():
        raise FileExistsError('Partial checkpoint exists; inspect before replacing')
    model.config.pathpred_cluster_manifest_sha256 = report['cluster_manifest_sha256']
    model.config.pathpred_dense_execution = report['execution_mode']
    model.config.pathpred_permutations_sha256 = perm_meta['sha256']
    model.save_pretrained(temp, safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(temp)
    shutil.copyfile(args.clusters / perm_meta['file'], temp / 'neuron-permutations.safetensors')
    reloaded = load_clustered_qwen(temp, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='eager', local_files_only=True)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), reloaded.parameters()))
    inputs = batches[0][1]
    with torch.inference_mode():
        expected_logits = model(**inputs, use_cache=False).logits
        reloaded_logits = reloaded(**inputs, use_cache=False).logits
        assert torch.equal(expected_logits, reloaded_logits)
        controller = gate_qwen3(reloaded, group_size=cm['fingerprint']['config']['group_size'])
        gates = {(l, k): torch.ones(n, device='cuda') for l in range(reloaded.config.num_hidden_layers)
                 for k, n in controller.units_per_layer.items()}
        with controller.use_gates(gates):
            assert torch.equal(expected_logits, reloaded(**inputs, use_cache=False).logits)
    report['checks']['checkpoint_reload_bitwise_equal'] = True
    report['checks']['reordered_all_on_gates_bitwise_equal'] = True
    report['checks']['embedding_lm_head_tying_preserved'] = model.lm_head.weight is model.model.embed_tokens.weight
    temp.rename(args.checkpoint)
    report['checkpoint_files'] = [{'file': f.name, 'bytes': f.stat().st_size, 'sha256': file_sha256(f)}
                                  for f in sorted(args.checkpoint.iterdir()) if f.is_file()]
    ccfg = cm['fingerprint']['config']
    cost = unit_costs(model.config, ccfg['group_size'], ccfg['cost_context_lengths'], ccfg['weight_bytes'])
    cost['model_revision'] = cfg['revision']
    cost['cluster_manifest_sha256'] = report['cluster_manifest_sha256']
    atomic_json(args.clusters / 'unit-costs.json', cost)
    report['unit_costs_sha256'] = file_sha256(args.clusters / 'unit-costs.json')
    report.update(status='passed', completed_at=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic() - started)
    atomic_json(args.output, report)
    print(json.dumps({'status': 'passed', 'seconds': report['seconds'], 'checks': report['checks']}), flush=True)


if __name__ == '__main__':
    main()
