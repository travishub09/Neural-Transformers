import torch
from transformers import MistralConfig, MistralForCausalLM
from pathpred.gated_qwen import gate_mistral
from pathpred.open_loop import PlannerConfig, SuffixPlanner
from pathpred.planner_training import end_to_end_batch
from pathpred.planner_decode import decode_batch
from pathpred.end_to_end import DualBudget


def setup():
    torch.manual_seed(123)
    c=MistralConfig(vocab_size=47,hidden_size=16,intermediate_size=32,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,sliding_window=64)
    c._attn_implementation='eager'
    model=MistralForCausalLM(c).eval()
    planner=SuffixPlanner(PlannerConfig(llm_layers=4,units=32,hidden_size=16,width=16,depth=2,heads=4))
    ids=torch.randint(3,47,(2,6))
    return model,planner,ids


def test_mistral_gate_equivalence_and_native_cache():
    model,p,ids=setup()
    with torch.no_grad():expected=model(ids).logits
    ctrl=gate_mistral(model)
    gates={(l,'mlp'):torch.ones(32) for l in range(4)}
    with torch.no_grad(),ctrl.use_gates(gates):
        actual=model(ids).logits
        cache=None;parts=[]
        for i in range(ids.shape[1]):
            out=model(ids[:,i:i+1],past_key_values=cache,use_cache=True)
            cache=out.past_key_values;parts.append(out.logits)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(torch.cat(parts,1),expected,rtol=1e-5,atol=1e-6)


def test_mistral_gradient_and_dense_execution_override():
    model,p,ids=setup();ctrl=gate_mistral(model);q=torch.ones(4,32)*.005
    before=p.output_weight.detach().clone()
    result=end_to_end_batch(model,ctrl,p,torch.optim.AdamW(p.parameters(),lr=.001),q,ids,
        torch.zeros(2),DualBudget(.5,.1,.05,100),mode='hard',chunk_size=4)
    assert result['whole_plan_calls']==1 and result['recomputations']>0
    assert not torch.equal(before,p.output_weight)
    assert all(not x.requires_grad and x.grad is None for x in model.parameters())
    hidden=torch.randn(2,6,16)
    plan=p.plan(hidden,mode='hard',offsets=torch.full((2,),100.),execution_prefix=3)
    assert plan.gates[:,:,0].eq(1).all() and plan.gates[:,:,1].eq(0).all()
    p.eval()
    value=decode_batch(model,ctrl,p,q,ids,torch.tensor([6,4]),offsets=torch.zeros(2),keep=.5,execution_prefix=3)
    assert value['layer_counts'].shape==(2,1,4)
    assert value['sums'][:,4].tolist()==[5,3]
