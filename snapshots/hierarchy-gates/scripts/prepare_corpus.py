"""Stream pinned public datasets into reproducible, document-isolated windows."""
import argparse
from collections import Counter, deque
import json
import os
import importlib.metadata
from pathlib import Path
import time

from datasets import load_dataset
from transformers import AutoTokenizer
import torch

from pathpred.collection import atomic_json, atomic_tensors, choose_pairs, document_hash, document_split, file_sha256


def content(row, source):
    if source["format"] == "messages":
        return "\n\n".join(f'{message["role"]}: {message["content"]}' for message in row[source["field"]])
    return row[source["field"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/collection.json"))
    p.add_argument("--model-config", type=Path, default=Path("configs/qwen3-1.7b-base.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    model_cfg = json.loads(args.model_config.read_text())
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Corpus already committed; reuse it rather than overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], revision=model_cfg["revision"], local_files_only=True)
    length = cfg["sequence_length"]
    streams = []
    for source in cfg["sources"]:
        stream = load_dataset(source["dataset"], name=source["config"], revision=source["revision"],
                              split=source["split"], streaming=True, token=False)
        streams.append(iter(stream.shuffle(seed=cfg["seed"], buffer_size=cfg["shuffle_buffer"])))
    queues = [deque() for _ in streams]
    used = [0] * len(streams)
    excluded = set()
    for corpus in cfg.get('exclude_corpora', []):
        for line in (Path(corpus)/'sequences.jsonl').read_text().splitlines():
            excluded.add(json.loads(line)['document_sha256'])
    seen = set(excluded)
    ids_list, masks, records = [], [], []
    started = time.monotonic()
    skipped_duplicates = 0
    while any(used[i] < source["tokens"] for i, source in enumerate(cfg["sources"])):
        i = min((i for i, s in enumerate(cfg["sources"]) if used[i] < s["tokens"]),
                key=lambda i: used[i] / cfg["sources"][i]["tokens"])
        source = cfg["sources"][i]
        if not queues[i]:
            row = next(streams[i])
            text = content(row, source).replace("\r\n", "\n").strip()
            if len(text) < 64:
                continue
            digest = document_hash(text)
            if digest in seen:
                skipped_duplicates += 1
                continue
            seen.add(digest)
            split = document_split(digest, cfg["split_salt"])
            if cfg.get('train_only', False) and split != 'train':
                continue
            tokens = tokenizer(text, add_special_tokens=False, truncation=True,
                               max_length=cfg["max_document_tokens"] - 1)["input_ids"]
            tokens.append(tokenizer.eos_token_id)
            provenance = {key: row[key] for key in ("id", "url", "repo_name", "path", "license", "prompt_id")
                          if key in row and isinstance(row[key], (str, int, float, bool))}
            for start in range(0, len(tokens), length):
                queues[i].append((tokens[start:start + length], digest, split, start, provenance))
        tokens, digest, split, start, provenance = queues[i].popleft()
        tokens = tokens[:source["tokens"] - used[i]]
        n = len(tokens)
        record = {"sequence_id": len(records), "document_sha256": digest, "split": split,
                  "domain": source["domain"], "source_index": i, "document_token_offset": start,
                  "length": n, "provenance": provenance}
        records.append(record)
        ids_list.append(tokens + [tokenizer.eos_token_id] * (length - n))
        masks.append([1] * n + [0] * (length - n))
        used[i] += n
        if len(records) % 1000 == 0:
            print(json.dumps({"sequences": len(records), "tokens_by_source": used}), flush=True)
    pairs = choose_pairs(records, cfg["sample_pairs"], cfg["seed"])
    tensors = atomic_tensors(args.output / "tokens.safetensors", {
        "input_ids": torch.tensor(ids_list, dtype=torch.int32),
        "attention_mask": torch.tensor(masks, dtype=torch.bool),
    })
    metadata = args.output / "sequences.jsonl"
    temp = metadata.with_suffix(".jsonl.tmp")
    with temp.open("w") as f:
        for record in records:
            record["sample_pair_start"] = pairs.get(record["sequence_id"])
            f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(metadata)
    counts = Counter()
    for row in records:
        counts[(row["domain"], row["split"])] += row["length"]
    manifest = {
        "schema_version": 1, "config": cfg, "model": model_cfg,
        "preparer_sha256": file_sha256(Path(__file__)),
        "sampling_source_sha256": file_sha256(Path("pathpred/collection.py")),
        "packages": {name: importlib.metadata.version(name) for name in ("transformers", "datasets", "tokenizers")},
        "tokens": sum(used), "sequences": len(records), "documents": len(set(r["document_sha256"] for r in records)),
        "sample_pairs": len(pairs), "skipped_exact_duplicates": skipped_duplicates,
        "excluded_document_hashes": len(excluded),
        "tokens_by_domain_split": {f'{k[0]}/{k[1]}': v for k, v in sorted(counts.items())},
        "files": [tensors, {"file": metadata.name, "sha256": file_sha256(metadata), "bytes": metadata.stat().st_size}],
        "elapsed_seconds": time.monotonic() - started,
        "sampling": "Bounded shuffled streaming prefixes; windows never cross documents. Up to one sampled adjacent pair per selected sequence; not uniform token sampling.",
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("config", "model")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
