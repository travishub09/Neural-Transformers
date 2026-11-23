"""Step 1: download a pinned base model and verify real CUDA inference.

This is a dense-model smoke test, not a gating or quality experiment.
Public model downloads do not require copying the user's HF token to the pod.
"""
import argparse
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; refusing a silent CPU fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The GPU must support bfloat16")
    torch.manual_seed(config["seed"])
    started = time.monotonic()
    print("Downloading pinned Qwen3 base weights to persistent cache", flush=True)
    model_path = snapshot_download(
        config["model_id"], revision=config["revision"], token=False,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation=config["attention_implementation"], local_files_only=True,
    ).eval().requires_grad_(False)
    actual = {key: getattr(model.config, key) for key in config["expected_architecture"]}
    if actual != config["expected_architecture"]:
        raise AssertionError(f"Unexpected model architecture: {actual}")
    inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        logits = model(**inputs).logits
        if not torch.isfinite(logits).all().item():
            raise AssertionError("Model produced non-finite logits")
        generated = model.generate(
            **inputs, max_new_tokens=24, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": config["model_id"], "revision": config["revision"],
        "architecture": actual, "dtype": str(next(model.parameters()).dtype),
        "attention_implementation": config["attention_implementation"],
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in (
            "transformers", "accelerate", "huggingface-hub", "safetensors")},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
        "logits_shape": list(logits.shape), "finite_logits": True,
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "setup_seconds": round(time.monotonic() - started, 2),
        "scope": "Dense smoke test only; no gated model or training implemented",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
