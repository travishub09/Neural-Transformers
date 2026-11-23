"""Small real Qwen3 models: no pretrained downloads needed for these tests."""
import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from pathpred import gate_qwen3


@pytest.fixture
def model():
    torch.manual_seed(123)
    torch.set_num_threads(2)
    config = Qwen3Config(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, attention_dropout=0.0,
        pad_token_id=0, eos_token_id=2, bos_token_id=1, tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval().requires_grad_(False)


@pytest.fixture
def tokens():
    return torch.tensor([[1, 5, 9, 12, 24, 8, 3], [1, 3, 17, 11, 6, 5, 4]])


def projection(model, layer, kind):
    block = model.model.layers[layer]
    return block.self_attn.o_proj if kind == "attention" else block.mlp.down_proj


def test_all_on_matches_logits_hidden_states_and_preserves_weights(model, tokens):
    with torch.no_grad():
        original = model(tokens, output_hidden_states=True, use_cache=False)
    params = dict(model.named_parameters())
    state_keys = set(model.state_dict())
    rng = torch.random.get_rng_state().clone()
    controller = gate_qwen3(model, group_size=16)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert set(model.state_dict()) == state_keys
    assert all(dict(model.named_parameters())[name] is value for name, value in params.items())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    gates = {(l, k): torch.ones(4) for l in range(3) for k in ("attention", "mlp")}
    with controller.use_gates(gates, record=True) as trace, torch.no_grad():
        actual = model(tokens, output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(original.logits, actual.logits, rtol=0, atol=0)
    for a, b in zip(original.hidden_states, actual.hidden_states):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(trace.records) == 6
    assert controller.units_per_layer == {"attention": 4, "mlp": 4}
    assert all(not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("kind", ["attention", "mlp"])
@pytest.mark.parametrize("mode", ["off", "single", "soft_per_token"])
def test_gates_match_independent_reference_hook(model, tokens, kind, mode):
    oracle = copy.deepcopy(model)
    controller = gate_qwen3(model, group_size=16)
    if mode == "off":
        gate = torch.zeros(4)
    elif mode == "single":
        gate = torch.tensor([1.0, 0.0, 1.0, 1.0])
    else:
        gate = torch.rand(2, 7, 4)
    target = projection(oracle, 1, kind)
    repeated = gate.repeat_interleave(target.in_features // 4, dim=-1)
    handle = target.register_forward_pre_hook(lambda module, args: (args[0] * repeated,))
    with torch.no_grad():
        expected = oracle(tokens, use_cache=False).logits
        with controller.use_gates({(1, kind): gate}, record=True) as trace:
            actual = model(tokens, use_cache=False).logits
    handle.remove()
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    record = trace.records[(1, kind)]
    assert record.gates.shape == (2, 7, 4)
    assert record.observed_norms.dtype == torch.float32
    assert not record.observed_norms.requires_grad
    assert torch.count_nonzero(record.observed_norms[~record.selected]) == 0
    assert torch.count_nonzero(record.contribution_norms[~record.selected]) == 0


@pytest.mark.parametrize("kind", ["attention", "mlp"])
def test_one_off_changes_only_target_layer_onward_and_target_batch_token(model, tokens, kind):
    with torch.no_grad():
        original = model(tokens, output_hidden_states=True, use_cache=False)
    controller = gate_qwen3(model, group_size=16)
    gate = torch.ones(2, 7, 4)
    gate[0, 3, 1] = 0
    with controller.use_gates({(1, kind): gate}), torch.no_grad():
        actual = model(tokens, output_hidden_states=True, use_cache=False)
    for idx in (0, 1):  # embedding and layer 0; layer 1 has not executed yet
        torch.testing.assert_close(original.hidden_states[idx], actual.hidden_states[idx], rtol=0, atol=0)
    assert not torch.equal(original.hidden_states[2][0, 3], actual.hidden_states[2][0, 3])
    torch.testing.assert_close(original.logits[1], actual.logits[1], rtol=0, atol=0)
    torch.testing.assert_close(original.logits[0, :3], actual.logits[0, :3], rtol=0, atol=0)


def test_activity_distinguishes_unselected_from_selected_zero(model):
    controller = gate_qwen3(model, group_size=16)
    proj = projection(model, 0, "attention")
    values = torch.zeros(1, 2, 32)
    values[..., 8:16] = 2
    with controller.use_gates({(0, "attention"): torch.tensor([1., 0., .5, 1.])}, record=True) as trace:
        proj(values)
    record = trace.records[(0, "attention")]
    assert record.selected[..., 0].all()  # a selected unit can genuinely have zero activity
    assert not record.selected[..., 1].any()
    assert record.observed_norms[..., 0].eq(0).all()
    assert record.observed_norms[..., 1].eq(0).all()  # don't leak the dense counterfactual norm


def test_soft_activity_is_pre_gate_and_contribution_is_post_gate(model):
    controller = gate_qwen3(model, group_size=16)
    x = torch.ones(1, 2, 64)
    with controller.use_gates({(0, "mlp"): torch.full((4,), .25)}, record=True) as trace:
        projection(model, 0, "mlp")(x)
    activity = trace.records[(0, "mlp")]
    torch.testing.assert_close(activity.observed_norms, torch.full((1, 2, 4), 4.))
    torch.testing.assert_close(activity.contribution_norms, torch.ones(1, 2, 4))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_loss_reaches_all_gate_tensors_not_frozen_weights(model, tokens, checkpointing):
    controller = gate_qwen3(model, group_size=16)
    if checkpointing:
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gates = {(l, k): torch.full((2, 7, 4), .8, requires_grad=True)
             for l in range(3) for k in ("attention", "mlp")}
    with controller.use_gates(gates):
        loss = model(tokens, labels=tokens, use_cache=False).loss
        loss.backward()
    assert torch.isfinite(loss)
    for g in gates.values():
        assert g.grad is not None and torch.isfinite(g.grad).all()
        assert torch.count_nonzero(g.grad) > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


def test_checkpointed_gradients_match_uncheckpointed(model, tokens):
    controller = gate_qwen3(model, group_size=16)
    outputs = []
    for checkpointing in (False, True):
        if checkpointing:
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        g = torch.full((4,), .7, requires_grad=True)
        with controller.use_gates({(0, "attention"): g, (2, "mlp"): g}):
            loss = model(tokens, labels=tokens, use_cache=False).loss
            loss.backward()
        outputs.append((loss.detach(), g.grad.clone()))
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)


def test_autograd_gate_gradient_matches_finite_difference(model, tokens):
    controller = gate_qwen3(model, group_size=16)
    g = torch.full((4,), .7, requires_grad=True)
    with controller.use_gates({(1, "mlp"): g}):
        loss = model(tokens, use_cache=False).logits.float().square().sum()
        loss.backward()
    eps = .01
    values = []
    for direction in (-1, 1):
        perturbed = g.detach().clone()
        perturbed[2] += direction * eps
        with controller.use_gates({(1, "mlp"): perturbed}), torch.no_grad():
            values.append(model(tokens, use_cache=False).logits.float().square().sum())
    numeric = (values[1] - values[0]) / (2 * eps)
    torch.testing.assert_close(g.grad[2], numeric, rtol=.02, atol=2e-4)


def test_optional_graph_connected_observations(model, tokens):
    controller = gate_qwen3(model, group_size=16)
    g = torch.full((4,), .8, requires_grad=True)
    with controller.use_gates({(0, "mlp"): g}, record=True, detach_observations=False) as trace:
        model(tokens, use_cache=False)
        trace.records[(1, "attention")].observed_norms.sum().backward()
    assert g.grad is not None and torch.count_nonzero(g.grad) > 0


def test_attention_gate_keeps_full_kv_cache(model, tokens):
    with torch.no_grad():
        original = model(tokens, use_cache=True).past_key_values
    controller = gate_qwen3(model, group_size=16)
    with controller.use_gates({(0, "attention"): torch.zeros(4)}), torch.no_grad():
        actual = model(tokens, use_cache=True).past_key_values
    # At the gated layer, Q/K/V are computed before gating, so cache is identical.
    # Later layer values may legitimately differ because their inputs changed.
    for expected, got in zip(original[0], actual[0]):
        torch.testing.assert_close(expected, got, rtol=0, atol=0)
    for k, v in actual:
        assert k.shape == v.shape == (2, 2, 7, 8)


@pytest.mark.parametrize("masked", [False, True])
def test_cached_chunks_match_full_sequence_with_same_gate_schedule(model, tokens, masked):
    controller = gate_qwen3(model, group_size=16)
    gates = {(l, k): torch.rand(2, 7, 4) if masked else torch.ones(2, 7, 4)
             for l in range(3) for k in ("attention", "mlp")}
    with controller.use_gates(gates), torch.no_grad():
        full = model(tokens, use_cache=False).logits
    cache = None
    pieces = []
    for start, end in ((0, 3), (3, 5), (5, 6), (6, 7)):
        chunk = {k: v[:, start:end] for k, v in gates.items()}
        with controller.use_gates(chunk), torch.no_grad():
            out = model(tokens[:, start:end], past_key_values=cache, use_cache=True,
                        attention_mask=torch.ones(2, end, dtype=torch.long))
        cache = out.past_key_values
        pieces.append(out.logits)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)
    assert cache.get_seq_length() == 7


def test_all_on_generate_and_padded_batch_match_original(model, tokens):
    mask = torch.ones_like(tokens)
    mask[1, :3] = 0
    tokens = tokens.clone()
    tokens[1, :3] = 0
    with torch.no_grad():
        expected = model.generate(tokens, attention_mask=mask, do_sample=False, max_new_tokens=5)
    controller = gate_qwen3(model, group_size=16)
    gates = {(l, k): torch.ones(4) for l in range(3) for k in ("attention", "mlp")}
    with controller.use_gates(gates), torch.no_grad():
        actual = model.generate(tokens, attention_mask=mask, do_sample=False, max_new_tokens=5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [
    {(9, "mlp"): torch.ones(4)}, {(0, "other"): torch.ones(4)},
    {(0, "mlp"): torch.ones(3)}, {(0, "mlp"): torch.ones(2, 4)},
    {(0, "mlp"): torch.tensor([0., 1., float("nan"), 0.])},
    {(0, "mlp"): torch.tensor([0., 1., float("inf"), 0.])},
    {(0, "mlp"): torch.full((4,), 1.1)}, {(0, "mlp"): torch.full((4,), -.1)},
    {(0, "mlp"): torch.ones(4, dtype=torch.long)},
    {(0, "mlp"): torch.empty(0, 1, 4)},
])
def test_bad_gates_rejected_before_mutation(model, bad):
    controller = gate_qwen3(model, group_size=16)
    with pytest.raises((ValueError, TypeError)):
        with controller.use_gates(bad):
            pytest.fail("Invalid gates accepted")
    assert not controller.state.active


def test_context_cleanup_on_failure_and_removal(model, tokens):
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits
    original = projection(model, 0, "attention")
    controller = gate_qwen3(model, group_size=16)
    with pytest.raises(ValueError, match="current batch/token"):
        with controller.use_gates({(0, "attention"): torch.zeros(2, 8, 4)}):
            model(tokens)
    with torch.no_grad():
        torch.testing.assert_close(model(tokens, use_cache=False).logits, expected, rtol=0, atol=0)
    with controller.use_gates():
        with pytest.raises(RuntimeError, match="Overlapping"):
            with controller.use_gates():
                pass
        with pytest.raises(RuntimeError, match="Exit"):
            controller.remove()
    controller.remove()
    assert projection(model, 0, "attention") is original
    with pytest.raises(RuntimeError, match="removed"):
        with controller.use_gates():
            pass


def test_reject_bad_installation_without_partial_mutation(model):
    original = projection(model, 0, "attention")
    with pytest.raises(ValueError):
        gate_qwen3(model, group_size=15)
    assert projection(model, 0, "attention") is original
    controller = gate_qwen3(model, group_size=16)
    with pytest.raises(TypeError, match="install twice"):
        gate_qwen3(model, group_size=16)
    controller.remove()


def test_explicit_broadcast_and_boolean_gates(model, tokens):
    controller = gate_qwen3(model, group_size=16)
    g = torch.tensor([True, False, True, False])
    results = []
    with torch.no_grad():
        for shape in (g, g[None, None], g[None, None].expand(2, 1, 4)):
            with controller.use_gates({(0, "mlp"): shape}):
                results.append(model(tokens, use_cache=False).logits)
    for actual in results[1:]:
        torch.testing.assert_close(actual, results[0], rtol=0, atol=0)


def test_recording_is_bounded_and_clearable(model, tokens):
    controller = gate_qwen3(model, group_size=16)
    with controller.use_gates(record=True) as trace, torch.no_grad():
        for _ in range(3):
            model(tokens, use_cache=False)
        assert len(trace.records) == 6
        trace.clear()
        assert trace.records == {}
