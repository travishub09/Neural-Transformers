"""Collect dense Qwen3 activity to atomic, checksummed, resumable shards."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download

from pathpred.collection import DenseRecorder, atomic_json, atomic_tensors, file_sha256, verify_shards


def estimated_bytes(tokens, samples, config):
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    neurons = config.intermediate_size
    hidden = config.hidden_size
    return (tokens * (layers * (heads + neurons // 128) * 2 + 16)
            + samples * (layers * neurons * 2 + (layers + 1) * hidden * 2 + layers * heads * 4 + 12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard-sequences", type=int, default=128)
    p.add_argument("--max-sequences", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-shards", type=int, help="Controlled interruption test; total committed shards")
    args = p.parse_args()
    if args.batch_size < 1 or args.shard_sequences < 1:
        raise ValueError("Batch and shard sizes must be positive")
    torch.set_num_threads(4)
    corpus = json.loads((args.corpus / "manifest.json").read_text())
    for f in corpus["files"]:
        if file_sha256(args.corpus / f["file"]) != f["sha256"]:
            raise ValueError("Corpus checksum mismatch")
    data = load_file(str(args.corpus / "tokens.safetensors"))
    records = [json.loads(line) for line in (args.corpus / "sequences.jsonl").read_text().splitlines()]
    limit = len(records) if args.max_sequences is None else min(args.max_sequences, len(records))
    if limit <= 0:
        raise ValueError("At least one sequence is required")
    records = records[:limit]
    pairs = {r["sequence_id"]: r["sample_pair_start"] for r in records if r["sample_pair_start"] is not None}
    cfg = corpus["model"]
    target_tokens = sum(r["length"] for r in records)
    target_samples = 2 * len(pairs)
    fingerprint = {
        "corpus_sha256": file_sha256(args.corpus / "manifest.json"),
        "collector_sha256": file_sha256(Path("pathpred/collection.py")),
        "runner_sha256": file_sha256(Path(__file__)), "model_revision": cfg["revision"],
        "max_sequences": limit, "batch_size": args.batch_size, "shard_sequences": args.shard_sequences,
        "norm_dtype": "float16", "raw_dtype": "bfloat16", "group_size": 128,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Use --resume to validate and continue the existing collection")
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Resume rejected: corpus, source code, or collection settings changed")
        verify_shards(args.output, manifest)
        if manifest["status"] == "complete":
            print("Collection already complete; all shard checksums verified", flush=True)
            return
    else:
        manifest = {"schema_version": 1, "fingerprint": fingerprint, "model": cfg,
                    "target_tokens": target_tokens, "target_samples": target_samples,
                    "target_sequences": limit, "status": "collecting", "shards": [],
                    "sequences_completed": 0, "tokens_completed": 0, "samples_completed": 0,
                    "collection_seconds": 0.0}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU fallback")
    torch.manual_seed(corpus["config"]["seed"])
    path = snapshot_download(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": "cuda:0"},
                                               attn_implementation="eager", local_files_only=True).eval().requires_grad_(False)
    remaining_bytes = estimated_bytes(target_tokens - manifest["tokens_completed"],
                                      target_samples - manifest["samples_completed"], model.config)
    free = shutil.disk_usage(args.output).free
    if remaining_bytes + 6 * 2**30 > free:
        raise RuntimeError(f"Need {remaining_bytes / 2**30:.2f} GiB plus 6 GiB reserve; only {free / 2**30:.2f} GiB free")
    manifest.update({"estimated_payload_bytes": estimated_bytes(target_tokens, target_samples, model.config),
                     "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                     "packages": {x: importlib.metadata.version(x) for x in ("transformers", "datasets", "safetensors")},
                     "attention_implementation": "eager", "status": "collecting"})
    atomic_json(manifest_path, manifest)
    recorder = DenseRecorder(model)
    try:
        for start in range(manifest["sequences_completed"], limit, args.shard_sequences):
            shard_started = time.monotonic()
            end = min(start + args.shard_sequences, limit)
            chunks = []
            for batch_start in range(start, end, args.batch_size):
                batch_end = min(batch_start + args.batch_size, end)
                ids = data["input_ids"][batch_start:batch_end].to("cuda", dtype=torch.long)
                mask = data["attention_mask"][batch_start:batch_end].to("cuda")
                chunks.append(recorder.collect(ids, mask, list(range(batch_start, batch_end)), pairs).tensors)
            tensors = {key: torch.cat([chunk[key] for chunk in chunks], dim=0) for key in chunks[0]}
            name = f"shard-{len(manifest['shards']):05d}.safetensors"
            if shutil.disk_usage(args.output).free < sum(t.numel() * t.element_size() for t in tensors.values()) + 6 * 2**30:
                raise RuntimeError("Storage reserve reached; prior committed shards are resumable")
            shard = atomic_tensors(args.output / name, tensors)
            shard.update({"sequence_start": start, "sequence_end": end,
                          "tokens": len(tensors["token_ids"]), "samples": len(tensors["sample_sequence_ids"]),
                          "seconds": time.monotonic() - shard_started})
            manifest["shards"].append(shard)
            manifest["sequences_completed"] = end
            manifest["tokens_completed"] += shard["tokens"]
            manifest["samples_completed"] += shard["samples"]
            manifest["collection_seconds"] += shard["seconds"]
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(manifest_path, manifest)
            print(json.dumps({"sequences": end, "tokens": manifest["tokens_completed"],
                              "samples": manifest["samples_completed"], "shards": len(manifest["shards"]),
                              "tokens_per_second": round(manifest["tokens_completed"] / manifest["collection_seconds"], 1)}), flush=True)
            del chunks, tensors
            if args.stop_after_shards and len(manifest["shards"]) >= args.stop_after_shards:
                manifest["status"] = "paused_for_resume_test"
                atomic_json(manifest_path, manifest)
                return
    finally:
        recorder.close()
    assert manifest["tokens_completed"] == target_tokens
    assert manifest["samples_completed"] == target_samples
    manifest["status"] = "complete"
    manifest["total_bytes"] = sum(s["bytes"] for s in manifest["shards"])
    atomic_json(manifest_path, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("shards", "model")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
