"""Fit balanced groups only on training activations; checkpoint every layer."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.clustering import activation_features, balanced_kmeans, grouping_metrics, validate_permutation


def read_samples(corpus, collection):
    manifest = json.loads((collection / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Collection must be complete")
    if file_sha256(corpus / "manifest.json") != manifest["fingerprint"]["corpus_sha256"]:
        raise ValueError("Wrong corpus manifest")
    cm = json.loads((corpus / "manifest.json").read_text())
    # Check sequence metadata before trusting its train/test identities.
    record_meta = next(v for v in cm["files"] if v["file"] == "sequences.jsonl")
    if file_sha256(corpus / "sequences.jsonl") != record_meta["sha256"]:
        raise ValueError("Sequence metadata checksum mismatch")
    records = [json.loads(x) for x in (corpus / "sequences.jsonl").read_text().splitlines()]
    document_splits = {}
    for r in records:
        if document_splits.setdefault(r["document_sha256"], r["split"]) != r["split"]:
            raise ValueError("Document split leakage")
    arch = manifest["model"]["expected_architecture"]
    total = manifest["target_samples"]
    # Layer-major RAM layout avoids repeatedly rereading all network-volume shards.
    raw = torch.empty((arch["num_hidden_layers"], total, arch["intermediate_size"]), dtype=torch.bfloat16)
    seqs, positions, splits = [], [], []
    offset = 0
    for i, shard in enumerate(manifest["shards"]):
        path = collection / shard["file"]
        if path.stat().st_size != shard["bytes"] or file_sha256(path) != shard["sha256"]:
            raise ValueError(f"Corrupt input {path.name}")
        with safe_open(str(path), framework="pt") as f:
            values = f.get_tensor("sample_mlp")
            seq = f.get_tensor("sample_sequence_ids")
            pos = f.get_tensor("sample_token_positions")
            expected = [(r["sequence_id"], p) for r in records[shard["sequence_start"]:shard["sequence_end"]]
                        if r["sample_pair_start"] is not None
                        for p in (r["sample_pair_start"], r["sample_pair_start"] + 1)]
            if list(zip(seq.tolist(), pos.tolist())) != expected:
                raise ValueError("Sample alignment mismatch")
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite raw activations")
            raw[:, offset:offset + len(seq)] = values.transpose(0, 1)
            seqs.extend(seq.tolist()); positions.extend(pos.tolist())
            splits.extend(records[s]["split"] for s in seq.tolist())
            offset += len(seq)
        if i % 10 == 0:
            print(json.dumps({"read_shards": i + 1, "total_shards": len(manifest["shards"])}), flush=True)
    if offset != total:
        raise ValueError("Incomplete samples")
    return raw, seqs, positions, splits, manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/clustering.json"))
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-layers", type=int)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg["fit_split"] != "train" or cfg["features"] != "absolute_activation_l2_normalized_per_neuron":
        raise ValueError("Unsupported feature or fit split")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("Full collection clustering requires CUDA")
    fingerprint = {"collection_manifest_sha256": file_sha256(args.collection / "manifest.json"),
                   "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.json"),
                   "config": cfg, "clustering_sha256": file_sha256(Path("pathpred/clustering.py")),
                   "runner_sha256": file_sha256(Path(__file__))}
    output_manifest = args.output / "manifest.json"
    if output_manifest.exists():
        report = json.loads(output_manifest.read_text())
        if not args.resume or report["fingerprint"] != fingerprint:
            raise ValueError("Existing output requires --resume and identical inputs/code/config")
        for item in report["layers"]:
            for entry in (item["tensors"], item["metrics_file"]):
                if file_sha256(args.output / entry["file"]) != entry["sha256"]:
                    raise ValueError("Corrupt completed layer")
        if report["status"] == "complete":
            for entry in report["outputs"].values():
                if file_sha256(args.output / entry["file"]) != entry["sha256"]:
                    raise ValueError("Corrupt completed output")
            print("Clustering already complete and checksummed", flush=True)
            return
    else:
        report = {"status": "in_progress", "fingerprint": fingerprint, "layers": [],
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "packages": {x: importlib.metadata.version(x) for x in ["torch", "ortools", "numpy", "safetensors"]}}
        atomic_json(output_manifest, report)
    raw, seqs, positions, splits, collection = read_samples(args.corpus, args.collection)
    report["samples_by_split"] = dict(Counter(splits))
    report["model"] = collection["model"]
    masks = {split: torch.tensor([s == split for s in splits]) for split in ("train", "validation", "test")}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Missing sample split")
    sample_meta = atomic_tensors(args.output / "sample-index.safetensors", {
        "sequence_ids": torch.tensor(seqs), "token_positions": torch.tensor(positions, dtype=torch.int32),
        "split_codes": torch.tensor([{"train": 0, "validation": 1, "test": 2}[s] for s in splits], dtype=torch.int32)})
    n = raw.shape[-1]
    for layer in range(len(report["layers"]), raw.shape[0]):
        started = time.monotonic()
        x = activation_features(raw[layer, masks["train"]].to("cuda"))
        perm, fit = balanced_kmeans(x, cfg["group_size"], seed=cfg["seed"] + layer * 100,
            **{k: cfg[k] for k in ["restarts", "max_iterations", "relative_tolerance", "distance_integer_scale"]})
        validate_permutation(perm, n)
        random_perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg["seed"] + 10000 + layer))
        metrics = {}
        for split, mask in masks.items():
            features = x if split == "train" else activation_features(raw[layer, mask].to("cuda"))
            metrics[split] = {name: grouping_metrics(features, order, cfg["group_size"])
                              for name, order in [("clustered", perm), ("original", torch.arange(n)), ("random", random_perm)]}
            if split != "train":
                del features
        del x
        # Norms for NEW groups; old full-corpus group norms remain unchanged.
        group_norms = torch.linalg.vector_norm(raw[layer].float()[:, perm].reshape(len(seqs), -1, cfg["group_size"]), dim=-1)
        entry = atomic_tensors(args.output / f"layer-{layer:02}.safetensors", {
            "permutation": perm, "inverse_permutation": torch.argsort(perm),
            "random_permutation": random_perm, "sample_group_norms": group_norms})
        details = {"layer": layer, "fit": fit, "metrics": metrics,
                   "seconds": time.monotonic() - started}
        metrics_path = args.output / f"layer-{layer:02}.json"
        atomic_json(metrics_path, details)
        report["layers"].append({"layer": layer, "tensors": entry,
            "metrics_file": {"file": metrics_path.name, "sha256": file_sha256(metrics_path)}, **details})
        atomic_json(output_manifest, report)
        print(json.dumps({"layer_complete": layer, "seconds": details["seconds"],
                          "train_objective": fit["objective"], "metrics": metrics}), flush=True)
        if args.stop_after_layers and len(report["layers"]) >= args.stop_after_layers:
            print("Controlled stop after committed layer", flush=True)
            return
    permutations = torch.stack([load_file(str(args.output / x["tensors"]["file"]))["permutation"] for x in report["layers"]])
    combined = atomic_tensors(args.output / "permutations.safetensors", {
        "permutations": permutations, "inverse_permutations": torch.argsort(permutations, dim=1)})
    report.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                  outputs={"sample_index": sample_meta, "permutations": combined})
    atomic_json(output_manifest, report)
    print(json.dumps({"status": "complete", "layers": len(report["layers"]), "samples": report["samples_by_split"]}), flush=True)


if __name__ == "__main__":
    main()
