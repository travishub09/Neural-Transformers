"""Structural checks and overhead measurements for an UNTRAINED path model.

No Phase 0 prediction-quality result, sparse route or optimizer update is run.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.path_model import PathConfig, PathModel
from pathpred.path_signals import censor_observations, previous_observations


def timed(function, repeats=10, warmup=3):
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        torch.cuda.synchronize()
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            function()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000)
    return {'median_ms': statistics.median(times), 'min_ms': min(times), 'max_ms': max(times),
            'samples_ms': times, 'warmup': warmup, 'repeats': repeats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--signals', type=Path, required=True)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--clusters', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    raw_config = json.loads(Path('configs/path-model.json').read_text())
    config = PathConfig(**raw_config['default'])
    torch.manual_seed(raw_config['seed'])
    sm = json.loads((args.signals / 'manifest.json').read_text())
    assert sm['status'] == 'complete'
    for entry in sm['files'].values():
        assert file_sha256(args.signals / entry['file']) == entry['sha256']
    assert sm['fingerprint']['signals_source_sha256'] == file_sha256(Path('pathpred/path_signals.py'))
    observations = load_file(str(args.signals / sm['files']['observations']['file']))
    # Observation censoring is applied to every sample before predecessor lookup.
    rng = torch.Generator().manual_seed(raw_config['seed'])
    keep = torch.rand(observations['binary'].shape, generator=rng) > .5
    censored = censor_observations(observations['binary'], keep)
    previous = previous_observations(censored, observations['previous_sample_indices'])
    selected = torch.where((observations['split_codes'] == 0) & (observations['previous_sample_indices'] >= 0))[0][:32]
    current = censored[selected].cuda()
    prev = previous[selected].cuda()
    valid = torch.ones(current.shape[:2], device='cuda', dtype=torch.bool)
    model = PathModel(config).cuda().eval()
    with torch.inference_mode():
        expected = model(current, previous=prev, previous_valid=valid)
        result, cache = [], None
        for layer in range(model.steps):
            prediction = model.step(current[:, layer], cache=cache, previous=prev[:, layer], previous_valid=valid[:, layer])
            assert prediction.target_layer == layer + config.lookahead
            result.append(prediction.logits); cache = prediction.cache
        actual = torch.stack(result, 1)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=4e-6)
        future, future_prev = current.clone(), prev.clone()
        future[:, 6:] = 1 - future[:, 6:].clamp_min(0)
        future_prev[:, 6:] = -1
        changed = model(future, previous=future_prev, previous_valid=valid)
        torch.testing.assert_close(changed[:, :6], expected[:, :6], rtol=0, atol=0)
    # Backward only: check trainability without changing any initialized weight.
    model.train()
    model(current[:4], previous=prev[:4], previous_valid=valid[:4]).square().mean().backward()
    gradients = {n: float(p.grad.float().norm()) for n, p in model.named_parameters() if p.grad is not None}
    assert len(gradients) == len(list(model.named_parameters()))
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True); model.eval()
    args.checkpoint.mkdir(parents=True, exist_ok=False)
    checkpoint_meta = atomic_tensors(args.checkpoint / 'initialized-path-model.safetensors',
                                     {n: v.detach().cpu() for n, v in model.state_dict().items()})
    atomic_json(args.checkpoint / 'config.json', {'model': asdict(config), 'seed': raw_config['seed'],
        'training_status': 'untrained_initialized', 'signal_manifest_sha256': file_sha256(args.signals / 'manifest.json')})
    restored = PathModel(config).cuda().eval()
    restored.load_state_dict(load_file(str(args.checkpoint / checkpoint_meta['file']), device='cuda'), strict=True)
    with torch.inference_mode():
        assert torch.equal(restored(current, previous=prev, previous_valid=valid), expected)
    del restored
    # Content features are after completed layer l (residual l+1), before target l+2.
    collection = json.loads((args.collection / 'manifest.json').read_text())
    first_shard = load_file(str(args.collection / collection['shards'][0]['file']))
    hidden = first_shard['sample_residuals'][:4, 1:].float().cuda()
    assert hidden.shape[1:] == (config.llm_layers, config.hidden_size)
    comparisons = {}
    unit_cost = json.loads((args.clusters / 'unit-costs.json').read_text())
    dense_flops = unit_cost['scenarios']['1024']['dense_accounted_flops']
    for signal in raw_config['comparisons']:
        variant = PathModel(replace(config, signal=signal)).cuda().eval()
        states = None if signal == 'hidden' else (censored[:4] if signal in ('binary', 'decisions')
                 else censor_observations(observations[signal][:4], keep[:4])).cuda()
        content = hidden if signal == 'hidden' else None
        with torch.inference_mode():
            output = variant(states, hidden=content)
        assert output.shape == (4, config.llm_layers - config.lookahead, config.units) and torch.isfinite(output).all()
        accounting = variant.accounting()
        accounting['fraction_of_dense_accounted_flops_at_1k'] = accounting['matmul_flops_per_llm_token'] / dense_flops
        comparisons[signal] = accounting
        del variant
    # Serial cached traversals reflect prototype overhead, not an optimized kernel.
    small_timing = None
    runtime = {}
    for name, cfg in [('default', config), ('original_proposal', replace(config, **raw_config['original_proposal']))]:
        predictor = model if name == 'default' else PathModel(cfg).cuda().eval()
        def traversal():
            cache = None
            for layer in range(predictor.steps):
                prediction = predictor.step(current[:1, layer], cache=cache,
                    previous=prev[:1, layer], previous_valid=valid[:1, layer])
                cache = prediction.cache
            return prediction.logits
        accounting = predictor.accounting()
        accounting['fraction_of_dense_accounted_flops_at_1k'] = accounting['matmul_flops_per_llm_token'] / dense_flops
        runtime[name] = {'accounting': accounting, 'cached_serial_traversal': timed(traversal),
                         'parallel_26_layer_call': timed(lambda: predictor(current[:1], previous=prev[:1], previous_valid=valid[:1]))}
        if name != 'default':
            del predictor
    assert runtime['default']['accounting']['fraction_of_dense_accounted_flops_at_1k'] < .01
    # Dense, unmodified pretrained reference: batch-one decode at fixed 1K cache.
    llm_cfg = collection['model']
    path = snapshot_download(llm_cfg['model_id'], revision=llm_cfg['revision'], local_files_only=True)
    llm = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='eager', local_files_only=True).eval().requires_grad_(False)
    context = 1024
    ids = torch.full((1, context), 42, dtype=torch.long, device='cuda')
    with torch.inference_mode():
        cache = llm(ids, use_cache=True, logits_to_keep=1).past_key_values
    token = ids[:, :1]
    def dense_decode():
        cache.crop(context)
        return llm(token, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    dense_timing = timed(dense_decode)
    for entry in runtime.values():
        entry['serial_controller_to_dense_decode_wall_ratio'] = entry['cached_serial_traversal']['median_ms'] / dense_timing['median_ms']
    report = {'status': 'passed', 'training_status': 'untrained_initialized',
              'completed_at': datetime.now(timezone.utc).isoformat(), 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'controller_dtype': 'float32', 'base_dtype': 'bfloat16',
              'source_sha256': {f: file_sha256(Path(f)) for f in
                  ['pathpred/path_model.py', 'pathpred/path_signals.py', 'pathpred/path_inputs.py', 'scripts/verify_path_model.py']},
              'signal_manifest_sha256': file_sha256(args.signals / 'manifest.json'),
              'config': raw_config, 'checkpoint': checkpoint_meta,
              'main_cached_vs_parallel_max_abs': float((actual - expected).abs().max()),
              'gradient_norms': gradients, 'comparisons_accounting': comparisons,
              'runtime': runtime, 'dense_decode_at_1k': dense_timing,
              'checks': ['input artifacts verified', 'masked current and previous observations',
                         'cached/parallel parity on real collected binary observations', 'future-layer causality',
                         'correct lookahead and dense first layers', 'all predictor parameters receive finite gradients',
                         'save/reload exact', 'all five input comparisons have finite outputs',
                         'default theoretical matmul cost under 1% of dense reference at 1K'],
              'limitations': ['No training updates, Phase 0 results, pruning-quality results or sparse execution',
                              'Controller serial benchmark uses pre-encoded observations and excludes LLM norm extraction',
                              'Python validation, synchronization and unfused launches remain; wall overhead is measured, not inferred from FLOPs',
                              'Parallel controller timing knows all layer observations in advance and is NOT an online latency claim',
                              'Base decode includes cache crop/reset; synthetic repeated token input measures runtime only',
                              'One-bit observations do not imply that weak units can safely be skipped',
                              'Comparison budgets must account for mandatory base computation and predictor overhead']}
    atomic_json(args.output, report)
    print(json.dumps({'status': report['status'], 'training_status': report['training_status'],
                      'parameters': model.accounting()['parameters'], 'runtime': runtime,
                      'dense_decode_at_1k': dense_timing}, indent=2), flush=True)


if __name__ == '__main__':
    main()
