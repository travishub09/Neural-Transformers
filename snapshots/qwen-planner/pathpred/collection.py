"""Dense activation recording and checksummed shard utilities for step 3."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def atomic_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(temp))
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    temp.replace(path)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_split(digest: str, salt: str) -> str:
    bucket = int(hashlib.sha256((salt + digest).encode()).hexdigest()[:16], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def choose_pairs(records: list[dict], count: int, seed: int) -> dict[int, int]:
    """Choose sequences without replacement, then one adjacent pair per sequence.

    Values are the FIRST position of the pair. This design samples sequences,
    not tokens uniformly; the manifest records that distinction explicitly.
    """
    ranked = []
    for row in records:
        if row["length"] < 2:
            continue
        key = f'{seed}:{row["document_sha256"]}:{row["document_token_offset"]}'
        digest = hashlib.sha256(key.encode()).digest()
        start = int.from_bytes(digest[8:16], "big") % (row["length"] - 1)
        ranked.append((digest[:8], row["sequence_id"], start))
    return {seq: start for _, seq, start in sorted(ranked)[:count]}


@dataclass
class DenseBatch:
    tensors: dict[str, torch.Tensor]


class DenseRecorder:
    """Read-only hooks on an untouched Qwen3 model, without gate instrumentation.

    Per-token norms exclude padding. Sampled raw SwiGLU intermediates use the
    original bf16 values. Residual states include inputs to all layers and the
    residual immediately before final normalization (L+1 states). No LM logits
    or teacher loss are collected in this stage.
    """

    def __init__(self, model, group_size: int = 128):
        self.model = model
        self.group_size = group_size
        self.layers = model.config.num_hidden_layers
        self.heads = model.config.num_attention_heads
        self.head_dim = model.config.head_dim
        self.neurons = model.config.intermediate_size
        if self.neurons % group_size:
            raise ValueError("MLP width must be divisible by group size")
        self.groups = self.neurons // group_size
        self.handles = []
        self.active = False
        self.values = {}
        for layer, block in enumerate(model.model.layers):
            self.handles.extend([
                block.register_forward_pre_hook(self._residual_hook(layer), with_kwargs=True),
                block.self_attn.o_proj.register_forward_pre_hook(self._projection_hook(layer, "head")),
                block.mlp.down_proj.register_forward_pre_hook(self._projection_hook(layer, "mlp")),
            ])
        self.handles.append(model.model.norm.register_forward_pre_hook(self._final_hook))

    def _residual_hook(self, layer):
        def hook(module, args, kwargs):
            if self.active:
                x = args[0] if args else kwargs["hidden_states"]
                self.values[("residual", layer)] = x[self.sample_batch, self.sample_position].detach().cpu()
        return hook

    def _final_hook(self, module, args):
        if self.active:
            self.values[("residual", self.layers)] = args[0][self.sample_batch, self.sample_position].detach().cpu()

    def _projection_hook(self, layer, kind):
        def hook(module, args):
            if not self.active:
                return
            x = args[0]
            units = self.heads if kind == "head" else self.groups
            width = self.head_dim if kind == "head" else self.group_size
            norms = torch.linalg.vector_norm(x.reshape(*x.shape[:2], units, width).float(), dim=-1)
            self.values[(kind, layer)] = norms[self.valid].to(torch.float16).cpu()
            if kind == "mlp":
                self.values[("neurons", layer)] = x[self.sample_batch, self.sample_position].detach().cpu()
            else:
                self.values[("sample_head", layer)] = norms[self.sample_batch, self.sample_position].cpu()
        return hook

    def collect(self, ids, attention_mask, sequence_ids, pair_starts) -> DenseBatch:
        if self.active:
            raise RuntimeError("Recorder is already active")
        self.valid = attention_mask.bool()
        selected = [(b, p) for b, seq in enumerate(sequence_ids) if seq in pair_starts
                    for p in (pair_starts[seq], pair_starts[seq] + 1)]
        self.sample_batch = torch.tensor([b for b, _ in selected], device=ids.device, dtype=torch.long)
        self.sample_position = torch.tensor([p for _, p in selected], device=ids.device, dtype=torch.long)
        if selected and not bool(self.valid[self.sample_batch, self.sample_position].all()):
            raise ValueError("Sampling padding or outside the current sequence is forbidden")
        self.values = {}
        self.active = True
        try:
            with torch.inference_mode():
                self.model.model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
            tensors = {
                "head_norms": torch.stack([self.values[("head", l)] for l in range(self.layers)], dim=1),
                "mlp_group_norms": torch.stack([self.values[("mlp", l)] for l in range(self.layers)], dim=1),
                "sample_mlp": torch.stack([self.values[("neurons", l)] for l in range(self.layers)], dim=1),
                "sample_head_norms": torch.stack([self.values[("sample_head", l)] for l in range(self.layers)], dim=1),
                "sample_residuals": torch.stack([self.values[("residual", l)] for l in range(self.layers + 1)], dim=1),
            }
            for name, value in tensors.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Non-finite values or FP16 norm overflow in {name}")
            seq = torch.tensor(sequence_ids, dtype=torch.int64)[:, None].expand(ids.shape).clone()
            positions = torch.arange(ids.shape[1], dtype=torch.int32)[None].expand(ids.shape)
            valid = self.valid.cpu()
            tensors.update({
                "token_ids": ids.detach().cpu()[valid].to(torch.int32),
                "sequence_ids": seq[valid], "token_positions": positions[valid],
                "sample_sequence_ids": torch.tensor([sequence_ids[b] for b, _ in selected], dtype=torch.int64),
                "sample_token_positions": self.sample_position.cpu().to(torch.int32),
            })
            return DenseBatch(tensors)
        finally:
            self.active = False
            self.values = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def verify_shards(directory: Path, manifest: dict) -> None:
    next_sequence = 0
    tokens = samples = 0
    for shard in manifest["shards"]:
        path = directory / shard["file"]
        if shard["sequence_start"] != next_sequence:
            raise ValueError("Non-contiguous shard ranges")
        if path.stat().st_size != shard["bytes"] or file_sha256(path) != shard["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {path.name}")
        next_sequence = shard["sequence_end"]
        tokens += shard["tokens"]
        samples += shard["samples"]
    if (next_sequence, tokens, samples) != (manifest["sequences_completed"], manifest["tokens_completed"], manifest["samples_completed"]):
        raise ValueError("Manifest totals disagree with committed shards")
