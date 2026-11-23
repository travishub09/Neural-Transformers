import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from pathpred import gate_qwen3
from pathpred.open_loop import PlannerConfig,SuffixPlanner,choose_gates
from pathpred.planner_route import planner_route,previous_context,record_parallel_paths
from pathpred.end_to_end import dense_teacher,quality_loss


def setup(autoregressive=True,previous_path=True):
    torch.set_num_threads(2);torch.manual_seed(41)
    p=SuffixPlanner(PlannerConfig(llm_layers=6,units=16,hidden_size=16,width=16,
        depth=2,heads=4,autoregressive=autoregressive,previous_path=previous_path)).eval()
    h=torch.randn(2,4,16);prev=torch.randint(-1,2,(2,4,6,16),dtype=torch.int8)
    valid=torch.ones(2,4,dtype=torch.bool);valid[:,0]=False
    labels=torch.rand(2,4,4,16)>.5
    return p,h,prev,valid,labels


def backbone():
    torch.manual_seed(17)
    c=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=16,num_hidden_layers=6,
                  num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    c._attn_implementation='eager';m=Qwen3ForCausalLM(c).eval()
    return m,gate_qwen3(m,1)


def test_teacher_forcing_is_causal_and_matches_incremental_forcing():
    p,h,prev,valid,labels=setup()
    expected=p.teacher_forced(h,labels,prev,valid)
    actual=p.plan(h,prev,valid,teacher_masks=labels,teacher_probability=1.)
    torch.testing.assert_close(expected,actual.scores,rtol=2e-5,atol=2e-6)
    changed=labels.clone();changed[:,:,2:]=~changed[:,:,2:]
    torch.testing.assert_close(p.teacher_forced(h,changed,prev,valid)[:,:,:3],expected[:,:,:3],rtol=0,atol=0)
    changed=labels.clone();changed[:,:,0]=~changed[:,:,0]
    changed_scores=p.teacher_forced(h,changed,prev,valid)
    assert torch.equal(changed_scores[:,:,0],expected[:,:,0])
    assert not torch.equal(changed_scores[:,:,1:],expected[:,:,1:])


def test_future_losses_reach_earlier_choice_heads_through_straight_through_chain():
    p,h,prev,valid,_=setup()
    plan=p.plan(h,prev,valid,mode='gumbel',temperature=.7)
    assert ((plan.gates==0)|(plan.gates==1)).all()
    plan.scores[:,:,-1].square().mean().backward()
    assert p.output_weight.grad[0].abs().sum()>0
    assert p.output_bias.grad[0].abs().sum()>0
    assert torch.isfinite(p.mask_tables.grad).all()


def test_one_shot_does_not_read_masks_and_previous_context_is_optional():
    for use_previous in (False,True):
        p,h,prev,valid,labels=setup(False,use_previous)
        inputs=(prev,valid) if use_previous else (None,None)
        a=p.teacher_forced(h,labels,*inputs);b=p.teacher_forced(h,~labels,*inputs)
        assert torch.equal(a,b)
        plan=p.plan(h,*inputs,mode='topk',keep=.5)
        assert (plan.gates.sum(-1)==8).all()
        assert p.mask_tables is None
        if use_previous:
            no_path=p.plan(h,prev,valid,drop_previous=True)
            absent=p.plan(h)
            assert torch.equal(no_path.scores,absent.scores)
        else:
            with pytest.raises(ValueError,match='Hidden-only'):p.plan(h,prev,valid)


def test_previous_context_shift_never_leaks_current_or_future_token():
    _,_,bits,_,_=setup();previous,valid=previous_context(bits)
    assert (previous[:,0]==-1).all() and not valid[:,0].any()
    assert torch.equal(previous[:,1:],bits[:,:-1])
    changed=bits.clone();changed[:,2:]=1
    changed_previous,_=previous_context(changed)
    assert torch.equal(previous[:,:3],changed_previous[:,:3])


def test_entire_plan_exists_before_suffix_and_no_suffix_observations_feed_back():
    p,_,_,_,_=setup();m,c=backbone();ids=torch.tensor([[2,4,6,8]])
    outputs=[];plans=[]
    for q in [torch.zeros(6,16),torch.full((6,16),100.)]:
        with torch.no_grad(),planner_route(m,c,p,q,mode='hard',checkpoint_layers=False) as trace:
            outputs.append(m(ids,use_cache=False).logits)
        events=trace['events'];boundary=events.index(('plan_complete',2))
        assert all(events.index(('execute',l))>boundary for l in range(2,6))
        assert trace['predictor_calls']==1
        assert all(trace['gates'][l].all() for l in (0,1))
        for l in range(6):assert (trace['bits'][l][trace['gates'][l].detach()==0]==-1).all()
        plans.append(trace['plan'].gates)
    assert torch.equal(outputs[0],outputs[1]) and torch.equal(plans[0],plans[1])


def test_all_on_reproduces_dense_qwen_and_quality_gradients_keep_backbone_frozen():
    p,_,_,_,_=setup();m,c=backbone();ids=torch.tensor([[2,4,6,8]]);q=torch.zeros(6,16)
    teacher,teacher_pre=dense_teacher(m,c,ids)
    with torch.no_grad(),planner_route(m,c,p,q,mode='dense',checkpoint_layers=False):
        assert torch.equal(m(ids,use_cache=False).logits,teacher)
    previous,valid=record_parallel_paths(m,c,p,q,ids)
    assert not previous.requires_grad and not valid[:,0].any()
    with planner_route(m,c,p,q,previous=previous,previous_valid=valid,mode='gumbel') as trace:
        student=m.model(ids,use_cache=False).last_hidden_state
        loss,_=quality_loss(m,student,trace['pre_norm'],teacher,teacher_pre,trace,chunk_size=2)
        loss.backward()
    assert trace['predictor_calls']==1 and trace['recomputations']>0
    assert p.output_weight.grad.abs().sum()>0 and p.mask_tables.grad.abs().sum()>0
    assert p.previous_tables.grad.abs().sum()>0
    assert all(not v.requires_grad and v.grad is None for v in m.parameters())


def test_gumbel_forward_is_binary_and_inference_is_deterministic():
    scores=torch.randn(3,16,requires_grad=True)
    a,_=choose_gates(scores,mode='gumbel');assert ((a==0)|(a==1)).all()
    a.sum().backward();assert scores.grad.abs().sum()>0
    a,_=choose_gates(scores,mode='hard');b,_=choose_gates(scores,mode='hard')
    assert torch.equal(a,b)


def test_cached_decode_batches_only_independent_documents_and_ignores_padding():
    from pathpred.planner_decode import decode_batch
    p,_,_,_,_=setup();m,c=backbone();q=torch.zeros(6,16)
    ids=torch.tensor([[2,4,6,8],[3,5,7,0]]);lengths=torch.tensor([4,3])
    kw={'offsets':torch.zeros(4),'keep':.5}
    joined=decode_batch(m,c,p,q,ids,lengths,**kw)
    assert joined['sums'][:,4].tolist()==[3.,2.]
    for i,n in enumerate(lengths):
        single=decode_batch(m,c,p,q,ids[i:i+1,:n],lengths[i:i+1],**kw)
        torch.testing.assert_close(joined['sums'][i],single['sums'][0],rtol=1e-5,atol=1e-6)
        assert torch.equal(joined['layer_counts'][i],single['layer_counts'][0])
    ids[1,-1]=17
    changed=decode_batch(m,c,p,q,ids,lengths,**kw)
    assert torch.equal(changed['sums'],joined['sums'])


def test_cached_retention_only_matches_quality_pass_and_ablation_is_causal():
    from pathpred.planner_decode import decode_batch
    p,_,_,_,_=setup();m,c=backbone();q=torch.zeros(6,16)
    ids=torch.tensor([[2,4,6,8]]);lengths=torch.tensor([4])
    kw={'offsets':torch.zeros(4),'keep':.5}
    a=decode_batch(m,c,p,q,ids,lengths,**kw)
    b=decode_batch(m,c,p,q,ids,lengths,quality=False,**kw)
    assert torch.equal(a['kept_sum'],b['kept_sum'])
    assert torch.equal(a['sums'][:,4],b['sums'][:,4])


def test_three_training_stages_update_only_planner_and_preserve_frozen_qwen():
    from pathpred.planner_training import copy_batch,end_to_end_batch
    from pathpred.end_to_end import DualBudget
    for ar,previous in [(True,True),(False,True),(False,False)]:
        p,_,_,_,_=setup(ar,previous);m,c=backbone();q=torch.zeros(6,16)
        ids=torch.tensor([[2,4,6,8]]);optimizer=torch.optim.AdamW(p.parameters(),lr=.001)
        before=p.output_weight.detach().clone()
        a=copy_batch(m,c,p,optimizer,q,ids,keep=.5)
        b=copy_batch(m,c,p,optimizer,q,ids,keep=.4,scheduled=True,teacher_probability=.5)
        d=DualBudget(.5,1.,.05,20.)
        z=end_to_end_batch(m,c,p,optimizer,q,ids,torch.zeros(4),d,chunk_size=2)
        assert all(torch.isfinite(torch.tensor(x['loss'])) for x in (a,b,z))
        assert not torch.equal(p.output_weight,before)
        assert z['whole_plan_calls']==1 and z['recomputations']>0
        assert all(v.grad is None and not v.requires_grad for v in m.parameters())
