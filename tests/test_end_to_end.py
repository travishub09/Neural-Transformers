import copy
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from pathpred import gate_qwen3
from pathpred.path_model import PathConfig
from pathpred.neuron_prediction import NeuronPredictor, route_neurons
from pathpred.end_to_end import (quality_route, dense_teacher, quality_loss,
                                neuron_gates, DualBudget)


def setup():
    torch.manual_seed(42); torch.set_num_threads(2)
    cfg = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=16,
                     num_hidden_layers=6, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=8)
    cfg._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(cfg).eval(); controller = gate_qwen3(model, 1)
    p = NeuronPredictor(PathConfig(llm_layers=6, units=16, width=16, depth=2,
        heads=4, hidden_size=16, hidden_summary=8, signal='binary')).eval()
    ids = torch.tensor([[2, 4, 6, 8], [3, 5, 7, 9]])
    q = torch.ones(6, 16) * .001
    return model, controller, p, ids, q


def test_dense_equivalence_and_exact_full_vocabulary_loss():
    m, c, p, ids, q = setup(); target, hidden = dense_teacher(m, c, ids)
    with quality_route(m, c, p, q, mode='dense') as trace:
        out = m.model(ids, use_cache=False).last_hidden_state
        assert torch.equal(m.lm_head(out), target)
        loss, stats = quality_loss(m, out, trace['pre_norm'], target, hidden, trace, alpha=0.)
        assert abs(float(loss)) < 1e-6
        assert stats['used'] == 1
    assert not c.state.active


def test_quality_gradient_reaches_predictor_but_never_backbone():
    for mode in ['soft', 'hard']:
        m, c, p, ids, q = setup(); before = {n: v.clone() for n, v in m.state_dict().items()}
        target, hidden = dense_teacher(m, c, ids)
        with quality_route(m, c, p, q, mode=mode) as trace:
            out = m.model(ids, use_cache=False).last_hidden_state
            loss, stats = quality_loss(m, out, trace['pre_norm'], target, hidden, trace, alpha=0.)
            loss.backward()
            assert trace['predictor_calls'] == 4
            assert trace['recomputations'] > 0
            assert p.output_weight.grad.abs().sum() > 0
            assert p.activity_projection.grad.abs().sum() > 0
            assert all(v.grad is None and not v.requires_grad for v in m.parameters())
            for l in range(6):
                mask = trace['gates'][l].detach() >= .5
                assert (trace['bits'][l][~mask] == -1).all()
                if l < 2: assert mask.all()
                if mode == 'hard': assert ((trace['gates'][l] == 0) | (trace['gates'][l] == 1)).all()
        torch.optim.SGD(p.parameters(), lr=.01).step()
        assert all(torch.equal(before[n], v) for n, v in m.state_dict().items())


def test_checkpoint_recomputation_matches_outputs_and_gradients():
    m, c, p, ids, q = setup(); target, hidden = dense_teacher(m, c, ids)
    results = []
    for checkpoint_layers in [False, True]:
        p.zero_grad(set_to_none=True)
        with quality_route(m, c, p, q, checkpoint_layers=checkpoint_layers) as trace:
            out = m.model(ids, use_cache=False).last_hidden_state
            loss, stats = quality_loss(m, out, trace['pre_norm'], target, hidden, trace,
                                      alpha=.3, checkpoint_head=checkpoint_layers)
            loss.backward()
            results.append((out.detach().clone(), {n: v.grad.clone() for n, v in p.named_parameters()}))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    for name in results[0][1]:
        torch.testing.assert_close(results[0][1][name], results[1][1][name], rtol=1e-5, atol=1e-7)


def test_hard_topk_matches_previous_guessing_router():
    m, c, p, ids, q = setup()
    with torch.no_grad(), c.use_gates(), route_neurons(m, c, q, .5, predictor=p):
        expected = m(ids, use_cache=False).logits
    with torch.no_grad(), quality_route(m, c, p, q, mode='topk', keep=.5):
        actual = m(ids, use_cache=False).logits
    assert torch.equal(expected, actual)


def test_dual_budget_and_straight_through_derivative():
    dual = DualBudget(target=.5, alpha=1., learning_rate=1.)
    assert dual.update(.8) == 1.3
    assert dual.update(.2) == 1.
    dual.alpha = 0.; assert dual.update(0.) == 0.
    x = torch.tensor([-1., 1.], requires_grad=True)
    gates, soft, observed = neuron_gates(x, mode='hard')
    assert gates.tolist() == [0., 1.] and observed.tolist() == [False, True]
    gates.sum().backward()
    torch.testing.assert_close(x.grad, soft.detach() * (1 - soft.detach()))


def test_hidden_baseline_matches_router_and_gets_quality_gradients():
    m,c,_,ids,q=setup()
    p=NeuronPredictor(PathConfig(llm_layers=6,units=16,width=16,depth=2,
        heads=4,hidden_size=16,hidden_summary=8,signal='hidden')).eval()
    with torch.no_grad(), c.use_gates(), route_neurons(m,c,q,.5,predictor=p):
        expected=m(ids,use_cache=False).logits
    with torch.no_grad(), quality_route(m,c,p,q,mode='topk',keep=.5):
        actual=m(ids,use_cache=False).logits
    assert torch.equal(expected,actual)
    target,hidden=dense_teacher(m,c,ids)
    with quality_route(m,c,p,q,mode='hard') as trace:
        out=m.model(ids,use_cache=False).last_hidden_state
        loss,_=quality_loss(m,out,trace['pre_norm'],target,hidden,trace,alpha=0.)
        loss.backward()
        assert p.output_weight.grad.abs().sum()>0
        assert any(v.grad is not None and v.grad.abs().sum()>0 for n,v in p.named_parameters() if 'hidden' in n)
        assert all(v.grad is None and not v.requires_grad for v in m.parameters())
