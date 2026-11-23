import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM
from pathpred import gate_qwen3
from pathpred.sparsity_diagnostics import block_gram, scores, select_mask, Condition, DynamicPolicy


def test_exact_contribution_matches_explicit_projected_blocks():
    torch.manual_seed(18)
    w=torch.randn(13,24);x=torch.randn(2,5,24);width=4
    calculated=scores(x,width,'contribution',w.norm(dim=0),block_gram(w,width))
    expected=torch.stack([(x[...,j:j+width]@w[:,j:j+width].T).square().sum(-1)
                          for j in range(0,24,width)],-1)
    torch.testing.assert_close(calculated,expected)


def test_neuron_contribution_is_wina_but_group_cross_terms_matter():
    w=torch.tensor([[1.,-1.],[0.,0.]])
    x=torch.tensor([[[1.,1.]]]);norm=w.norm(dim=0)
    assert torch.equal(scores(x,1,'wina',norm),scores(x,1,'contribution',norm))
    assert scores(x,2,'contribution',norm,block_gram(w,2)).item()==0
    assert scores(x,2,'wina',norm).item()==2


def test_protected_units_count_inside_exact_budget():
    x=torch.arange(20.).reshape(2,10)
    mask=select_mask(x,.5,torch.tensor([0,1]))
    assert (mask.sum(-1)==5).all() and mask[:,:2].all()
    assert not mask[:,2:5].any()
    with pytest.raises(ValueError):select_mask(x,.1,torch.tensor([0,1]))


def test_dynamic_policy_matches_direct_hook_and_preserves_inactive_component():
    torch.manual_seed(9)
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=6,
                   num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    cfg._attn_implementation='eager'
    m=Qwen3ForCausalLM(cfg).eval().requires_grad_(False)
    ids=torch.tensor([[1,8,9,4,22,3]])
    p=DynamicPolicy(m);c=Condition('mlp',16,'magnitude',.5,'edges')
    with torch.inference_mode(),p.apply(c,capture=True):reference=m(ids,use_cache=False).logits
    assert all(mask.all() for (l,k),mask in p.latest_masks.items() if k=='attention' or l in [0,1,4,5])
    assert (p.latest_masks[2,'mlp'].sum(-1)==2).all()
    ctl=gate_qwen3(m,1)
    with torch.inference_mode(),ctl.use_gates(),p.apply(c,ctl):actual=m(ids,use_cache=False).logits
    assert torch.equal(actual,reference)
    assert p.fractions()['attention']==1
    assert p.fractions()['mlp']==pytest.approx(5/6)
    ctl.remove()
