from dataclasses import replace
import pytest
import torch
from pathpred.path_signals import UnitThresholds, censor_observations, preceding_sample_indices, previous_observations
from pathpred.path_model import PathConfig, PathModel


def small(**kwargs):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    return PathModel(PathConfig(llm_layers=6, units=4, width=16, depth=2, heads=4,
                               hidden_size=12, hidden_summary=4, **kwargs))


def observations():
    rng = torch.Generator().manual_seed(71)
    return torch.randint(-1, 2, (3, 6, 4), generator=rng, dtype=torch.int8)


def test_thresholds_use_only_training_and_strict_above_own_quantile():
    values = torch.arange(8.).reshape(8, 1, 1).expand(-1, 2, 3).clone()
    values[:, :, 1] *= 100
    train = torch.tensor([True] * 4 + [False] * 4)
    q = UnitThresholds.fit(values, train)
    assert q.upper[0].tolist() == [2.25, 225., 2.25]
    values[~train] = float('nan')
    assert torch.equal(UnitThresholds.fit(values, train).upper, q.upper)
    at = q.upper.expand(2, -1, -1).clone()
    ran = torch.ones_like(at, dtype=torch.bool)
    at[1] += 1
    ran[0, 0, 0] = False
    at[0, 0, 0] = float('nan')
    codes = q.encode(at, ran)
    assert codes.dtype == torch.int8 and codes[0, 0, 0] == -1
    assert (codes[0, 1] == 0).all() and (codes[1] == 1).all()


def test_main_signal_cannot_reveal_magnitude_or_gradient_and_hides_skipped_units():
    q = UnitThresholds(torch.ones(6, 4), torch.full((6, 4), .25))
    ran = torch.rand(3, 6, 4) > .4
    a = torch.full((3, 6, 4), 2., requires_grad=True)
    b = torch.full((3, 6, 4), 2000.)
    b[~ran] = float('nan')
    codes_a, codes_b = q.encode(a, ran), q.encode(b, ran)
    assert torch.equal(codes_a, codes_b) and not codes_a.requires_grad
    model = small()
    out = model(codes_a)
    assert torch.equal(out, model(codes_b))
    out.square().mean().backward()
    assert a.grad is None and any(p.grad is not None for p in model.parameters())
    with pytest.raises(TypeError):
        model(a)
    with pytest.raises(ValueError, match='forbidden'):
        model(codes_a, hidden=torch.ones(3, 6, 12))


def test_unknown_is_distinct_from_observed_weak_and_masking_is_applied():
    model = small()
    off = torch.zeros(3, 6, 4, dtype=torch.int8)
    missing = torch.full_like(off, -1)
    assert not torch.equal(model(off), model(missing))
    keep = torch.zeros_like(off, dtype=torch.bool)
    assert torch.equal(censor_observations(off, keep), missing)


def test_sample_predecessors_do_not_cross_gaps_or_windows():
    seq = torch.tensor([0, 0, 0, 0, 1, 1, 2])
    pos = torch.tensor([4, 5, 8, 9, 0, 1, 9])
    prev = preceding_sample_indices(seq, pos)
    assert prev.tolist() == [-1, 0, -1, 2, -1, 4, -1]
    states = torch.arange(7).reshape(7, 1, 1).expand(-1, 6, 4)
    actual = previous_observations(states, prev)
    assert actual[:, 0, 0].tolist() == [-1, 0, -1, 2, -1, 4, -1]
    with pytest.raises(ValueError):
        preceding_sample_indices(torch.tensor([0, 0]), torch.tensor([1, 1]))


def test_future_current_and_previous_layer_activity_cannot_change_past_predictions():
    model = small().eval()
    states, previous = observations(), observations().flip(0)
    valid = torch.ones(3, 6, dtype=torch.bool)
    baseline = model(states, previous=previous, previous_valid=valid)
    later, later_previous = states.clone(), previous.clone()
    later[:, 2:] = 1 - later[:, 2:].clamp_min(0)
    later_previous[:, 2:] = -1
    changed = model(later, previous=later_previous, previous_valid=valid)
    torch.testing.assert_close(baseline[:, :2], changed[:, :2], rtol=0, atol=0)
    assert model.target_layers == (2, 3, 4, 5)
    assert model.initial_dense_mask(3).shape == (3, 2, 4)
    assert model.initial_dense_mask(3).all()


@pytest.mark.parametrize('signal', ['binary', 'decisions', 'coarse', 'magnitudes', 'hidden', 'hybrid'])
def test_cached_and_parallel_paths_match_including_previous_token(signal):
    model = small(signal=signal).eval()
    states = observations()
    if signal == 'coarse':
        states = states.clone(); states[states == 1] = 2
    if signal == 'magnitudes':
        states = states.float()
    hidden = torch.randn(3, 6, 12) if signal in ('hidden', 'hybrid') else None
    if signal == 'hidden':
        states = None
    previous = None if states is None else states.flip(0)
    ph = None if hidden is None else hidden.flip(0)
    valid = torch.ones(3, 6, dtype=torch.bool)
    valid[0] = False
    expected = model(states, previous=previous, hidden=hidden, previous_hidden=ph, previous_valid=valid)
    cache, result = None, []
    for layer in range(model.steps):
        part = lambda x: None if x is None else x[:, layer]
        prediction = model.step(part(states), cache=cache, previous=part(previous), hidden=part(hidden),
                                previous_hidden=part(ph), previous_valid=valid[:, layer])
        assert prediction.target_layer == layer + 2
        assert prediction.cache.next_layer == layer + 1
        assert all(k.shape[2] == layer + 1 for k, v in prediction.cache.blocks)
        result.append(prediction.logits); cache = prediction.cache
    torch.testing.assert_close(torch.stack(result, 1), expected, rtol=3e-5, atol=3e-6)
    with pytest.raises(ValueError, match='reset'):
        model.step(part(states), cache=cache, hidden=part(hidden))


def test_decisions_only_has_no_access_to_observed_strong_bits():
    model = small(signal='decisions')
    a = observations()
    b = torch.where(a < 0, a, 1 - a)
    assert torch.equal(model(a), model(b))


def test_previous_token_unavailable_values_cannot_leak():
    model = small(signal='hybrid').eval()
    states = observations(); hidden = torch.randn(3, 6, 12)
    valid = torch.zeros(3, 6, dtype=torch.bool)
    a = model(states, previous=states, hidden=hidden, previous_hidden=hidden, previous_valid=valid)
    b = model(states, previous=torch.full_like(states, 100), hidden=hidden,
              previous_hidden=torch.full_like(hidden, float('nan')), previous_valid=valid)
    assert torch.equal(a, b)


def test_incremental_cache_supports_training_gradients_and_rejects_other_model():
    first = small(); second = small()
    states = observations()
    first(states).square().mean().backward()
    cache, results = None, []
    for layer in range(second.steps):
        prediction = second.step(states[:, layer], cache=cache)
        results.append(prediction.logits); cache = prediction.cache
    torch.stack(results, 1).square().mean().backward()
    for (name, p), (_, q) in zip(first.named_parameters(), second.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=3e-6, msg=name)
    own_cache = second.step(states[:, 0]).cache
    with pytest.raises(ValueError, match='different'):
        first.step(states[:, 1], cache=own_cache)


def test_layer_and_unit_permutation_orbit_is_not_silently_mixed():
    model = small()
    states = observations()
    assert not torch.equal(model(states), model(states.flip(-1)))
    assert not torch.equal(model(states), model(states.flip(1)))


def test_cost_accounting_and_smaller_default_are_explicit():
    tiny = PathModel(PathConfig())
    large = PathModel(replace(PathConfig(), width=256, depth=4))
    assert tiny.accounting()['parameters'] == sum(p.numel() for p in tiny.parameters())
    assert tiny.accounting()['matmul_flops_per_llm_token'] == 23_788_544
    assert large.accounting()['matmul_flops_per_llm_token'] > 7 * tiny.accounting()['matmul_flops_per_llm_token']
    assert tiny.accounting()['cache_elements_per_llm_token'] == 2 * 2 * 26 * 128


def test_no_previous_token_ablation_and_invalid_codes():
    model = small(previous_token=False)
    states = observations()
    assert model(states).shape == (3, 4, 4)
    with pytest.raises(ValueError, match='disabled'):
        model(states, previous=states)
    with pytest.raises(ValueError, match='code'):
        model(torch.full_like(states, 2))


def test_online_trace_adapter_exposes_only_selected_binary_observations():
    from pathpred.gated_qwen import ActivityTrace, UnitActivity
    from pathpred.path_inputs import encode_trace_layer
    thresholds = UnitThresholds(torch.ones(6, 4), torch.zeros(6, 4))
    trace = ActivityTrace()
    for kind in ('attention', 'mlp'):
        norms = torch.tensor([[[.5, 2.]]])
        selected = torch.tensor([[[True, False]]])
        trace.records[(0, kind)] = UnitActivity(selected.float(), selected, norms, norms)
    codes = encode_trace_layer(thresholds, trace, 0)
    assert codes.tolist() == [[[0, -1, 0, -1]]]
