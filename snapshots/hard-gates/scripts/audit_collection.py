"""Read every committed shard and verify alignment, finiteness, and sample norms."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, file_sha256, verify_shards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.collection / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Collection is not complete")
    verify_shards(args.collection, manifest)
    assert file_sha256(args.corpus / "manifest.json") == manifest["fingerprint"]["corpus_sha256"]
    corpus = load_file(str(args.corpus / "tokens.safetensors"))
    records = [json.loads(x) for x in (args.corpus / "sequences.jsonl").read_text().splitlines()]
    splits = {}
    counts = Counter()
    sample_counts = Counter()
    for row in records[:manifest["target_sequences"]]:
        prior = splits.setdefault(row["document_sha256"], row["split"])
        assert prior == row["split"], "Document split leakage"
        counts[(row["domain"], row["split"])] += row["length"]
        if row["sample_pair_start"] is not None:
            sample_counts[(row["domain"], row["split"])] += 2
    checked_tokens = checked_samples = 0
    worst_head_relative_error = 0.0
    length = corpus["input_ids"].shape[1]
    architecture = manifest["model"]["expected_architecture"]
    layers, heads = architecture["num_hidden_layers"], architecture["num_attention_heads"]
    neurons, hidden = architecture["intermediate_size"], architecture["hidden_size"]
    for shard in manifest["shards"]:
        t = load_file(str(args.collection / shard["file"]))
        nt, ns = shard["tokens"], shard["samples"]
        for key, shape, dtype in (
            ("head_norms", (nt, layers, heads), torch.float16),
            ("mlp_group_norms", (nt, layers, neurons // 128), torch.float16),
            ("sample_mlp", (ns, layers, neurons), torch.bfloat16),
            ("sample_head_norms", (ns, layers, heads), torch.float32),
            ("sample_residuals", (ns, layers + 1, hidden), torch.bfloat16),
        ):
            assert tuple(t[key].shape) == shape and t[key].dtype == dtype
        for value in t.values():
            assert torch.isfinite(value).all()
        start, end = shard["sequence_start"], shard["sequence_end"]
        mask = corpus["attention_mask"][start:end]
        expected_seq = torch.arange(start, end)[:, None].expand(mask.shape)[mask]
        expected_pos = torch.arange(length, dtype=torch.int32)[None].expand(mask.shape)[mask]
        assert torch.equal(t["sequence_ids"], expected_seq)
        assert torch.equal(t["token_positions"], expected_pos)
        assert torch.equal(t["token_ids"], corpus["input_ids"][start:end][mask])
        sample_pairs = [(r["sequence_id"], pos) for r in records[start:end] if r["sample_pair_start"] is not None
                        for pos in (r["sample_pair_start"], r["sample_pair_start"] + 1)]
        assert t["sample_sequence_ids"].tolist() == [s for s, _ in sample_pairs]
        assert t["sample_token_positions"].tolist() == [p for _, p in sample_pairs]
        token_keys = t["sequence_ids"] * length + t["token_positions"]
        sample_keys = t["sample_sequence_ids"] * length + t["sample_token_positions"]
        indices = torch.searchsorted(token_keys, sample_keys)
        assert torch.equal(token_keys[indices], sample_keys)
        assert torch.equal(t["head_norms"][indices], t["sample_head_norms"].half())
        if len(indices):
            raw = t["sample_mlp"].float()
            norms = torch.linalg.vector_norm(raw.reshape(raw.shape[0], raw.shape[1], -1, 128), dim=-1)
            torch.testing.assert_close(t["mlp_group_norms"][indices].float(), norms, rtol=.001, atol=1e-5)
            relative = (t["head_norms"][indices].float() - t["sample_head_norms"]).abs() / t["sample_head_norms"].clamp_min(1e-6)
            worst_head_relative_error = max(worst_head_relative_error, relative.max().item())
        checked_tokens += len(t["token_ids"])
        checked_samples += len(t["sample_sequence_ids"])
    assert checked_tokens == manifest["target_tokens"]
    assert checked_samples == manifest["target_samples"]
    result = {"status": "passed", "shards": len(manifest["shards"]), "tokens": checked_tokens,
              "sample_positions": checked_samples, "document_split_overlap": 0,
              "tokens_by_domain_split": {f'{a}/{b}': n for (a, b), n in sorted(counts.items())},
              "sample_positions_by_domain_split": {f'{a}/{b}': n for (a, b), n in sorted(sample_counts.items())},
              "worst_sampled_head_norm_storage_relative_error": worst_head_relative_error,
              "collection_manifest_sha256": file_sha256(args.collection / "manifest.json"),
              "checks": ["all shard checksums", "finite stored values", "token and sample alignment", "no document split overlap",
                         "sampled MLP norms recomputed from raw activations", "sampled head norm quantization"]}
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
