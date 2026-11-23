"""Verify step 2 on the pinned pretrained model; no training or dataset collection.

Run from the repository root: python -m scripts.verify_gated_model
The independent reference masks untouched projection inputs with temporary hooks.
"""
import argparse
import copy
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathpred import gate_qwen3


def difference(a, b):
    delta = (a.float() - b.float()).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "exact": torch.equal(a, b)}


def exact(a, b):
    result = difference(a, b)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qwen3-1.7b-base.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/step2-verification.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("The pretrained verification requires CUDA")
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    path = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="eager", local_files_only=True,
    ).eval().requires_grad_(False)
    original = copy.deepcopy(model)
    parameters = {name: (p, p._version) for name, p in model.named_parameters()}
    state_keys = set(model.state_dict())
    controller = gate_qwen3(model, group_size=128)
    assert set(model.state_dict()) == state_keys
    assert all(dict(model.named_parameters())[name] is p for name, (p, _) in parameters.items())
    gates = {(layer, kind): torch.ones(units, device="cuda")
             for layer in range(model.config.num_hidden_layers)
             for kind, units in controller.units_per_layer.items()}
    report = {
        "model_id": cfg["model_id"], "revision": cfg["revision"],
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "python": platform.python_version(), "dtype": "bfloat16",
        "attention_implementation": "eager", "group_size": 128,
        "layers": len(model.model.layers), "units_per_layer": controller.units_per_layer,
        "source_sha256": hashlib.sha256(Path("pathpred/gated_qwen.py").read_bytes()).hexdigest(),
        "all_on": {}, "masked_reference": {},
    }
    examples = {
        "prose": ["The capital of France is Paris. Explain why cities grow near rivers."],
        "code_and_math": ["def add(a, b):\n    return a + b\n", "If x + 7 = 19, then x ="],
        "padded_batch": ["Hello.", "A frozen language model uses attention and feed-forward layers to predict the next token."],
        "longer_context": ["The small model reads the passage and predicts the next word. " * 24],
    }
    for name, texts in examples.items():
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            baseline = original(**inputs, use_cache=False).logits
            with controller.use_gates(gates, record=True) as trace:
                actual = model(**inputs, use_cache=False).logits
        report["all_on"][name] = {**exact(baseline, actual), "shape": list(actual.shape)}
        assert len(trace.records) == 56
        for (layer, kind), activity in trace.records.items():
            assert activity.gates.shape == (*inputs.input_ids.shape, controller.units_per_layer[kind])
            assert activity.selected.all()
            assert torch.isfinite(activity.observed_norms).all()
        del baseline, actual, trace
        print(f"All-on exact match: {name}", flush=True)

    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        expected_tokens = original.generate(**inputs, do_sample=False, max_new_tokens=12,
                                            pad_token_id=tokenizer.pad_token_id)
        with controller.use_gates(gates):
            actual_tokens = model.generate(**inputs, do_sample=False, max_new_tokens=12,
                                          pad_token_id=tokenizer.pad_token_id)
    assert torch.equal(expected_tokens, actual_tokens)
    report["all_on_generation_identical"] = True

    # Both implementations see exactly the same prefill and successive decode calls.
    baseline_cache = actual_cache = None
    cache_checks = []
    for start, end in ((0, 4), (4, 7), (7, inputs.input_ids.shape[1])):
        ids = inputs.input_ids[:, start:end]
        with torch.inference_mode():
            baseline = original(ids, past_key_values=baseline_cache, use_cache=True)
            with controller.use_gates(gates):
                actual = model(ids, past_key_values=actual_cache, use_cache=True)
        cache_checks.append(exact(baseline.logits, actual.logits))
        baseline_cache, actual_cache = baseline.past_key_values, actual.past_key_values
        for expected_kv, actual_kv in zip(baseline_cache, actual_cache):
            for a, b in zip(expected_kv, actual_kv):
                assert torch.equal(a, b)
    report["all_on_cached_chunks"] = cache_checks
    del baseline_cache, actual_cache, baseline, actual

    for kind, width in (("attention", model.config.head_dim), ("mlp", 128)):
        units = controller.units_per_layer[kind]
        mask = torch.ones(1, inputs.input_ids.shape[1], units, device="cuda")
        mask[:, 3:, 0] = 0
        mask[:, 5:, 1] = .5
        parent = original.model.layers[12]
        proj = parent.self_attn.o_proj if kind == "attention" else parent.mlp.down_proj
        hook = proj.register_forward_pre_hook(
            lambda module, values, mask=mask, width=width:
                (values[0] * mask.to(values[0].dtype).repeat_interleave(width, dim=-1),)
        )
        with torch.inference_mode():
            expected = original(**inputs, use_cache=True, output_hidden_states=True)
            with controller.use_gates({(12, kind): mask}, record=True) as trace:
                actual = model(**inputs, use_cache=True, output_hidden_states=True)
        hook.remove()
        report["masked_reference"][kind] = exact(expected.logits, actual.logits)
        activity = trace.records[(12, kind)]
        assert not activity.selected[0, 3:, 0].any()
        assert activity.observed_norms[0, 3:, 0].eq(0).all()
        assert activity.contribution_norms[0, 3:, 0].eq(0).all()
        # Prefill with gates, then decode one more token without gates: past KV
        # must remain correct and available to future token queries.
        next_token = torch.tensor([[42]], device="cuda")
        with torch.inference_mode():
            expected_next = original(next_token, past_key_values=expected.past_key_values, use_cache=True)
            actual_next = model(next_token, past_key_values=actual.past_key_values, use_cache=True)
        report["masked_reference"][kind + "_next_token"] = exact(expected_next.logits, actual_next.logits)
        for k, v in actual_next.past_key_values:
            assert k.shape[1] == v.shape[1] == model.config.num_key_value_heads
        del expected, actual, expected_next, actual_next, trace
        print(f"Independent masked reference and subsequent cache match: {kind}", flush=True)

    # Compare gradients with and without recomputation on the full frozen model.
    gradients = []
    losses = []
    for checkpointing in (False, True):
        if checkpointing:
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        train_gates = {key: torch.full_like(value, .8, dtype=torch.float32, requires_grad=True)
                       for key, value in gates.items()}
        with controller.use_gates(train_gates):
            loss = model(**inputs, labels=inputs.input_ids, use_cache=False).loss
            loss.backward()
        assert torch.isfinite(loss)
        for gate in train_gates.values():
            assert gate.grad is not None and torch.isfinite(gate.grad).all()
            assert torch.count_nonzero(gate.grad) > 0
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        gradients.append(torch.cat([g.grad.flatten() for g in train_gates.values()]))
        losses.append(loss.item())
        del loss, train_gates
    report["gate_gradients"] = {
        "losses": losses, "elements": gradients[0].numel(),
        "checkpoint_comparison": exact(gradients[0], gradients[1]),
        "all_56_gate_tensors_finite_and_nonzero": True,
        "base_parameter_gradients_absent": True,
    }
    model.gradient_checkpointing_disable()
    model.eval()
    assert all(p._version == initial_version for p, initial_version in parameters.values())
    report["original_parameters_unchanged"] = True
    report["peak_cuda_allocated_gib_bf16_tests"] = torch.cuda.max_memory_allocated() / 2**30
    del gradients, gates, parameters
    # A short FP32 check avoids treating bf16 rounding as an equivalence allowance.
    model.float()
    original.float()
    fp32_gates = {(l, k): torch.ones(u, device="cuda") for l in range(28)
                  for k, u in controller.units_per_layer.items()}
    with torch.inference_mode():
        baseline = original(**inputs, use_cache=False).logits
        with controller.use_gates(fp32_gates):
            actual = model(**inputs, use_cache=False).logits
    report["all_on_float32"] = exact(baseline, actual)
    report["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["scope"] = "Dense gate simulation only; no training, quality-retention, or speedup claim"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
