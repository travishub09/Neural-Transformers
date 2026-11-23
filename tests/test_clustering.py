import copy
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from pathpred import gate_qwen3
from pathpred.clustering import (activation_features, balanced_assignment, balanced_kmeans,
                                 grouping_metrics, reorder_mlp_, unit_costs, validate_permutation)


def test_balanced_assignment_matches_exhaustive_optimum():
    rng = np.random.default_rng(8)
    d = rng.integers(0, 100, size=(6, 3)).astype(float)
    labels = balanced_assignment(d, 2).numpy()
    costs = [sum(d[i, a] for i, a in enumerate(assign))
             for assign in itertools.product(range(3), repeat=6)
             if all(assign.count(j) == 2 for j in range(3))]
    assert d[np.arange(6), labels].sum() == min(costs)
    assert np.array_equal(np.bincount(labels), [2, 2, 2])
    with pytest.raises(ValueError):
        balanced_assignment(d, 3)
    with pytest.raises(ValueError):
        balanced_assignment(np.full((6, 3), np.nan), 2)


def test_clustering_recovers_coactive_groups_and_is_deterministic():
    raw = torch.tensor([[1., 0., 1., 0.], [2., 0., 2., 0.],
                        [0., -1., 0., 1.], [0., 3., 0., -3.]])
    x = activation_features(raw)
    p, info = balanced_kmeans(x, 2, seed=42, restarts=2, max_iterations=10)
    again, _ = balanced_kmeans(x, 2, seed=42, restarts=2, max_iterations=10)
    assert torch.equal(p, again)
    assert {frozenset(v.tolist()) for v in p.reshape(2, 2)} == {frozenset([0, 2]), frozenset([1, 3])}
    assert info["objective"] < 1e-10
    assert grouping_metrics(x, p, 2)["mean_pairwise_magnitude_cosine"] == pytest.approx(1, abs=1e-6)
    assert grouping_metrics(x, torch.arange(4), 2)["mean_pairwise_magnitude_cosine"] == 0
    assert torch.allclose(x, activation_features(raw * torch.tensor([3., 2., 8., 1.])))


def test_zero_activity_features_remain_finite_and_balanced():
    x = activation_features(torch.zeros(3, 8))
    p, result = balanced_kmeans(x, 2, seed=3, max_iterations=3)
    validate_permutation(p, 8)
    assert result["objective"] == 0
    assert grouping_metrics(x, p, 2)["zero_activity_neurons"] == 8
    with pytest.raises(ValueError):
        activation_features(torch.full((2, 3), float('inf')))


class BiasedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(5, 12)
        self.up_proj = nn.Linear(5, 12)
        self.down_proj = nn.Linear(12, 5)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def test_reordering_handles_biases_and_inverse_restores_every_weight():
    torch.manual_seed(7)
    mlp = BiasedMLP().double()
    state = {n: p.clone() for n, p in mlp.state_dict().items()}
    ids = {n: id(p) for n, p in mlp.named_parameters()}
    p = torch.randperm(12)
    x = torch.randn(2, 4, 5, dtype=torch.float64)
    expected = mlp(x)
    reorder_mlp_(mlp, p)
    torch.testing.assert_close(mlp(x), expected, rtol=1e-13, atol=1e-13)
    assert ids == {n: id(v) for n, v in mlp.named_parameters()}
    reorder_mlp_(mlp, torch.argsort(p))
    assert all(torch.equal(v, mlp.state_dict()[n]) for n, v in state.items())
    with pytest.raises(ValueError):
        reorder_mlp_(mlp, torch.zeros(12, dtype=torch.long))


def test_reordered_qwen_logits_cache_and_masked_group_semantics():
    torch.manual_seed(9)
    torch.set_num_threads(2)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0., tie_word_embeddings=True)
    config._attn_implementation = 'eager'
    original = Qwen3ForCausalLM(config).double().eval().requires_grad_(False)
    model = copy.deepcopy(original)
    permutations = [torch.randperm(64) for _ in range(3)]
    for block, p in zip(model.model.layers, permutations):
        reorder_mlp_(block.mlp, p)
    ids = torch.tensor([[1, 4, 9, 6], [2, 5, 7, 8]])
    with torch.inference_mode():
        a = original(ids, use_cache=True)
        b = model(ids, use_cache=True)
        torch.testing.assert_close(a.logits, b.logits, rtol=1e-11, atol=1e-11)
        an = original(ids[:, :1], past_key_values=a.past_key_values, use_cache=True)
        bn = model(ids[:, :1], past_key_values=b.past_key_values, use_cache=True)
        torch.testing.assert_close(an.logits, bn.logits, rtol=1e-11, atol=1e-11)
    controller = gate_qwen3(model, group_size=16)
    mask = torch.tensor([0., 1., .5, 1.], dtype=torch.float64)
    old_mask = torch.empty(64, dtype=torch.float64)
    old_mask[permutations[1]] = mask.repeat_interleave(16)
    handle = original.model.layers[1].mlp.down_proj.register_forward_pre_hook(lambda m, args: (args[0] * old_mask,))
    with torch.inference_mode(), controller.use_gates({(1, 'mlp'): mask}):
        torch.testing.assert_close(original(ids).logits, model(ids).logits, rtol=1e-11, atol=1e-11)
    handle.remove()


@pytest.mark.parametrize('q,kv,h,m,g', [(16, 8, 2048, 6144, 128), (32, 8, 4096, 12288, 192)])
def test_cost_vector_sums_to_independent_dense_matmul_accounting(q, kv, h, m, g):
    config = SimpleNamespace(num_attention_heads=q, num_key_value_heads=kv, head_dim=128,
        hidden_size=h, intermediate_size=m, num_hidden_layers=28, vocab_size=151936)
    costs = unit_costs(config, g)
    assert len(costs['per_layer_unit_weight_bytes']) == q + m // g
    for text, scenario in costs['scenarios'].items():
        context = int(text)
        per_layer = 2 * h * (q + 2 * kv) * 128 + 2 * q * 128 * h + 6 * h * m + 4 * q * context * 128
        dense = 28 * per_layer + 2 * h * config.vocab_size
        assert scenario['dense_accounted_flops'] == dense
        assert scenario['mandatory_fraction_of_accounted_flops'] > 0
    assert costs['query_head_to_kv_head'][-1] == kv - 1
