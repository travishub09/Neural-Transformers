import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from pathpred.collection import (
    DenseRecorder, atomic_json, atomic_tensors, choose_pairs,
    document_hash, document_split, verify_shards,
)


@pytest.fixture
def dense_model():
    torch.manual_seed(73)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=8, max_position_embeddings=64, attention_dropout=0.0)
    cfg._attn_implementation = "eager"
    return Qwen3ForCausalLM(cfg).eval().requires_grad_(False)


def test_dense_recorder_alignment_padding_and_exact_model_output(dense_model):
    ids = torch.tensor([[1, 4, 7, 9, 3], [1, 6, 8, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    with torch.no_grad():
        baseline_output = dense_model(ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
        baseline = baseline_output.logits
    recorder = DenseRecorder(dense_model)
    independent = {}
    block = dense_model.model.layers[0]

    def oracle_mlp(module, args):
        x = args[0]
        independent["mlp"] = (module.act_fn(module.gate_proj(x)) * module.up_proj(x)).detach()

    handle = block.mlp.register_forward_pre_hook(oracle_mlp)
    result = recorder.collect(ids, mask, [100, 101], {100: 2, 101: 0}).tensors
    handle.remove()
    with torch.no_grad():
        actual = dense_model(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert result["head_norms"].shape == (8, 2, 4)
    assert result["mlp_group_norms"].shape == (8, 2, 2)
    assert result["sample_mlp"].shape == (4, 2, 256)
    assert result["sample_residuals"].shape == (4, 3, 32)
    assert result["sample_sequence_ids"].tolist() == [100, 100, 101, 101]
    assert result["sample_token_positions"].tolist() == [2, 3, 0, 1]
    assert result["sequence_ids"].tolist() == [100] * 5 + [101] * 3
    assert result["token_positions"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    assert result["token_ids"].tolist() == [1, 4, 7, 9, 3, 1, 6, 8]
    expected = independent["mlp"][[0, 0, 1, 1], [2, 3, 0, 1]]
    torch.testing.assert_close(result["sample_mlp"][:, 0], expected, rtol=0, atol=0)
    for layer in range(2):
        expected_hidden = baseline_output.hidden_states[layer][[0, 0, 1, 1], [2, 3, 0, 1]]
        torch.testing.assert_close(result["sample_residuals"][:, layer], expected_hidden, rtol=0, atol=0)
    final = dense_model.model.norm(result["sample_residuals"][:, -1])
    torch.testing.assert_close(final, baseline_output.hidden_states[-1][[0, 0, 1, 1], [2, 3, 0, 1]], rtol=0, atol=0)
    indices = torch.tensor([2, 3, 5, 6])
    norm = torch.linalg.vector_norm(result["sample_mlp"].reshape(4, 2, 2, 128), dim=-1)
    torch.testing.assert_close(result["mlp_group_norms"][indices], norm.half(), rtol=0, atol=0)
    assert torch.equal(result["head_norms"][indices], result["sample_head_norms"].half())
    assert all(not t.requires_grad for t in result.values())
    recorder.close()
    assert all(len(m._forward_pre_hooks) == 0 for m in dense_model.modules())


def test_dense_recorder_empty_sampling_and_rejects_padding(dense_model):
    recorder = DenseRecorder(dense_model)
    ids = torch.tensor([[1, 2, 3]])
    result = recorder.collect(ids, torch.ones_like(ids), [0], {}).tensors
    assert result["sample_mlp"].shape == (0, 2, 256)
    assert result["sample_residuals"].shape == (0, 3, 32)
    with pytest.raises(ValueError, match="padding"):
        recorder.collect(ids, torch.tensor([[1, 1, 0]]), [0], {0: 1})
    assert not recorder.active
    recorder.close()


def test_recorder_restores_active_flag_after_model_error(dense_model):
    recorder = DenseRecorder(dense_model)
    with pytest.raises((IndexError, RuntimeError)):
        recorder.collect(torch.tensor([[9999, 9999]]), torch.ones(1, 2), [0], {})
    assert not recorder.active and recorder.values == {}
    recorder.close()


def test_pairs_are_deterministic_adjacent_and_sequence_sampled():
    rows = [{"sequence_id": i, "document_sha256": str(i // 2),
             "document_token_offset": i % 2 * 512, "length": 5 + i} for i in range(100)]
    a = choose_pairs(rows, 20, 17)
    assert a == choose_pairs(list(reversed(rows)), 20, 17)
    assert len(a) == 20
    assert all(0 <= pos < rows[seq]["length"] - 1 for seq, pos in a.items())
    assert a != choose_pairs(rows, 20, 18)
    assert choose_pairs([dict(rows[0], length=1)], 20, 17) == {}


def test_document_split_is_stable_across_chunks_and_sources():
    digest = document_hash("same document")
    assert document_split(digest, "salt") == document_split(document_hash("same document"), "salt")
    assert document_hash("different document") != digest
    splits = {document_split(document_hash(str(i)), "salt") for i in range(1000)}
    assert splits == {"train", "validation", "test"}


def test_atomic_shards_and_resume_reject_corruption_and_gaps(tmp_path):
    info = atomic_tensors(tmp_path / "shard-00000.safetensors", {"x": torch.ones(2, 3)})
    assert torch.equal(load_file(str(tmp_path / info["file"]))["x"], torch.ones(2, 3))
    info.update(sequence_start=0, sequence_end=2, tokens=5, samples=2)
    manifest = {"shards": [info], "sequences_completed": 2, "tokens_completed": 5, "samples_completed": 2}
    atomic_json(tmp_path / "manifest.json", manifest)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    verify_shards(tmp_path, manifest)
    bad = dict(manifest, tokens_completed=6)
    with pytest.raises(ValueError, match="totals"):
        verify_shards(tmp_path, bad)
    info["sequence_start"] = 1
    with pytest.raises(ValueError, match="ranges"):
        verify_shards(tmp_path, manifest)
    info["sequence_start"] = 0
    with (tmp_path / info["file"]).open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_shards(tmp_path, manifest)
