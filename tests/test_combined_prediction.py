import pytest
import torch
from pathpred.combined_prediction import CombinedNeuronPredictor
from pathpred.end_to_end import quality_route, dense_teacher, quality_loss
from pathpred.neuron_prediction import NeuronPredictor, route_neurons, stack_stream
from pathpred.path_model import PathConfig
from pathpred import gate_qwen3
from transformers import Qwen3Config, Qwen3ForCausalLM


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    cfg = PathConfig(llm_layers=6, units=16, width=16, depth=2, heads=4,
                     hidden_size=16, hidden_summary=8)
    path = NeuronPredictor(cfg).eval()
    combined = CombinedNeuronPredictor(cfg).eval()
    combined.load_path_state(path.state_dict())
    states = torch.randint(-1, 2, (2, 5, 6, 16), dtype=torch.int8)
    hidden = torch.randn(2, 5, 6, 16)
    return path, combined, states, hidden


def backbone():
    torch.manual_seed(17)
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=16,
        num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    cfg._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(cfg).eval()
    return model, gate_qwen3(model, 1)


def test_zero_projection_is_exact_path_and_can_learn():
    path, p, states, hidden = setup()
    expected = path.forward_tokens(states=states)
    actual = p.forward_tokens(states=states, hidden=hidden)
    assert torch.equal(expected, actual)
    actual.square().mean().backward()
    assert p.hidden_input_weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='parameter names'):
        p.load_path_state({})


def test_combined_is_causal_and_incremental_matches_batch():
    _, p, states, hidden = setup()
    torch.nn.init.normal_(p.hidden_input_weight, std=.1)
    full = p.forward_tokens(states=states, hidden=hidden)
    previous = None
    outputs = []
    for token in range(states.shape[1]):
        cache = None
        next_embeddings = {}
        layers = []
        for layer in range(p.steps):
            result = p.step_tokens(states=states[:, token:token+1, layer],
                hidden=hidden[:, token:token+1, layer], cache=cache,
                previous_embedding=None if previous is None else previous[layer],
                embedding_store=next_embeddings)
            assert result.target_layer == layer + 2
            cache = result.cache
            layers.append(result.logits)
        outputs.append(torch.stack(layers, 2))
        previous = next_embeddings
    torch.testing.assert_close(torch.cat(outputs, 1), full, rtol=2e-5, atol=2e-6)
    changed_states, changed_hidden = states.clone(), hidden.clone()
    changed_states[:, 3:] = 0
    changed_hidden[:, 3:] = 30
    changed_states[:, :, 3:] = 0
    changed_hidden[:, :, 3:] = -20
    changed = p.forward_tokens(states=changed_states, hidden=changed_hidden)
    assert torch.equal(changed[:, :3, :3], full[:, :3, :3])
    # Earlier tokens' hidden states cannot enter the next token's path embedding.
    changed_hidden = hidden.clone()
    changed_hidden[:, 0] = 100
    changed = p.forward_tokens(states=states, hidden=changed_hidden)
    assert torch.equal(changed[:, 1:], full[:, 1:])


def test_zero_warmstart_preserves_closed_loop_masks_logits_and_observations():
    path, p, _, _ = setup()
    model, controller = backbone()
    ids = torch.tensor([[2, 4, 6, 8]])
    thresholds = torch.full((6, 16), .001)
    with torch.no_grad(), quality_route(model, controller, path, thresholds,
            mode='hard', checkpoint_layers=False) as first:
        expected = model(ids, use_cache=False).logits
    with torch.no_grad(), quality_route(model, controller, p, thresholds,
            mode='hard', checkpoint_layers=False) as second:
        actual = model(ids, use_cache=False).logits
    assert torch.equal(actual, expected)
    for layer in range(6):
        assert torch.equal(first['gates'][layer], second['gates'][layer])
        assert torch.equal(first['bits'][layer], second['bits'][layer])
        if layer < 2:
            assert second['gates'][layer].all()
        assert (second['bits'][layer][second['gates'][layer] == 0] == -1).all()


def test_quality_gradients_reach_new_projection_with_frozen_backbone():
    _, p, _, _ = setup()
    model, controller = backbone()
    ids = torch.tensor([[2, 4, 6, 8]])
    thresholds = torch.full((6, 16), .001)
    teacher, teacher_pre = dense_teacher(model, controller, ids)
    with quality_route(model, controller, p, thresholds, mode='hard') as trace:
        student = model.model(ids, use_cache=False).last_hidden_state
        loss, _ = quality_loss(model, student, trace['pre_norm'], teacher, teacher_pre,
                              trace, alpha=1., chunk_size=2)
        loss.backward()
    assert torch.isfinite(p.hidden_input_weight.grad).all()
    assert p.hidden_input_weight.grad.abs().sum() > 0
    assert all(not v.requires_grad and v.grad is None for v in model.parameters())
    assert trace['predictor_calls'] == 4
    assert trace['recomputations'] > 0


def test_reliance_modes_remove_only_documented_inputs_and_restore():
    path, p, states, hidden = setup()
    torch.nn.init.normal_(p.hidden_input_weight, std=.1)
    with p.reliance('no_hidden'):
        assert torch.equal(p.forward_tokens(states=states, hidden=hidden), path.forward_tokens(states=states))
    zeroed = torch.where(states >= 0, 0, states)
    expected = p.forward_tokens(states=zeroed, hidden=hidden)
    with p.reliance('zero_bits'):
        assert torch.equal(expected, p.forward_tokens(states=states, hidden=hidden))
    with p.reliance('no_path'):
        assert torch.equal(p.forward_tokens(states=states, hidden=hidden),
                           p.forward_tokens(states=torch.zeros_like(states), hidden=hidden))
    assert p.reliance_mode == 'none'
    with pytest.raises(RuntimeError), p.reliance('no_hidden'):
        raise RuntimeError('test restore')
    assert p.reliance_mode == 'none'


def test_accounting_counts_one_current_hidden_projection():
    path, p, _, _ = setup()
    a, b = path.accounting(), p.accounting()
    extra = p.steps * p.config.hidden_size * p.config.width
    assert b['parameters'] - a['parameters'] == extra
    assert b['matmul_flops_per_llm_token'] - a['matmul_flops_per_llm_token'] == 2 * extra


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA BF16 warm-start equivalence')
def test_bf16_warmstart_is_exact():
    path, p, states, hidden = setup()
    path.cuda(); p.cuda(); states = states.cuda(); hidden = hidden.cuda()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        assert torch.equal(path.forward_tokens(states=states), p.forward_tokens(states=states, hidden=hidden))
