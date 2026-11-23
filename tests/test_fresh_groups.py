import torch
from pathpred.fresh_groups import original_to_stored_order, contribution_mask, contribution_gram


def test_original_to_stored_mapping_and_scattered_mask():
    torch.manual_seed(7)
    stored=torch.randperm(24);fresh=torch.randperm(24)
    p=original_to_stored_order(fresh,stored)
    assert torch.equal(stored[p],fresh)
    x=torch.randn(2,5,24);w=torch.randn(11,24)
    gram=contribution_gram(w,p,4)
    mask=contribution_mask(x,p,4,.5,gram)
    explicit=torch.stack([(x[...,idx]@w[:,idx].T).square().sum(-1)
                          for idx in p.reshape(-1,4)],-1)
    chosen=explicit.topk(3,dim=-1).indices
    reference=torch.zeros_like(mask)
    for b in range(2):
        for t in range(5):
            reference[b,t,p.reshape(-1,4)[chosen[b,t]].flatten()]=True
    assert torch.equal(mask,reference)
    assert (mask.sum(-1)==12).all()
    assert contribution_mask(x,p,4,1.,gram).all()


def test_identity_grouping_reproduces_existing_policy():
    from pathpred.sparsity_diagnostics import block_gram,scores,select_mask
    torch.manual_seed(11)
    x=torch.randn(1,9,32);w=torch.randn(7,32);p=torch.arange(32)
    expected=select_mask(scores(x,4,'contribution',None,block_gram(w,4)),.75).repeat_interleave(4,-1)
    actual=contribution_mask(x,p,4,.75,contribution_gram(w,p,4))
    assert torch.equal(actual,expected)
