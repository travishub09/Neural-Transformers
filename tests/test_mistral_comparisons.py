import pytest,torch
from torch.nn import functional as F
from tests.test_mistral_planner import setup
from pathpred.gated_qwen import gate_mistral
from scripts.mistral_comparison_core import Policy,score_batch,griffin_scores,attention_sketch


def test_dense_suffix_likelihood_matches_full_sequence_and_resets():
    model,p,ids=setup();ctrl=gate_mistral(model)
    seq=[ids[0].tolist(),ids[1].tolist()];prompts=[2,3]
    pol=Policy(model,ctrl,'dense')
    got=score_batch(pol,seq,prompts)
    with torch.no_grad():lp=F.log_softmax(model(ids).logits.float(),-1)
    for i,n in enumerate(prompts):
        expected=-lp[i,n-1:-1].gather(-1,ids[i,n:,None]).sum()
        assert got['sums'][i,1]==pytest.approx(expected.item(),abs=1e-5)
        assert got['sums'][i,4]==len(seq[i])-n
    again=score_batch(pol,seq,prompts)
    torch.testing.assert_close(got['sums'],again['sums'],rtol=0,atol=0)


def test_griffin_uses_only_prompt_and_exact_sparse_budget():
    model,p,ids=setup();ctrl=gate_mistral(model)
    pol=Policy(model,ctrl,'griffin',prefix=2,keep=.5)
    got=score_batch(pol,[ids[0].tolist()],[2]);masks={k:v.clone() for k,v in pol.fixed.items()}
    assert got['executions'].item()==len(ids[0])-2-1
    assert got['sums'][0,4].item()==len(ids[0])-2
    changed=ids[0].tolist();changed[3:]=[4,5,6]
    score_batch(pol,[changed],[2])
    for k,v in masks.items():assert torch.equal(v,pol.fixed[k])
    assert (got['kept_sum']/got['executions']).item()==.5
    expected=torch.tensor([[1.,0.]])
    torch.testing.assert_close(griffin_scores(torch.tensor([[[2.,0.],[0.,9.]]]),torch.tensor([[1,0]])),expected)


def test_planner_dense_override_and_previous_path_isolation():
    model,p,ids=setup();ctrl=gate_mistral(model);q=torch.ones(4,32)*.005;p.eval()
    pol=Policy(model,ctrl,'planner',prefix=2,planner=p,q=q,offsets=torch.full((2,),-100.))
    got=score_batch(pol,[ids[0].tolist(),ids[1].tolist()],[2,3])
    assert got['sums'][:,2].abs().max()<1e-6
    assert (got['kept_sum']/got['executions']).eq(1).all()
    for i in range(2):
        one=score_batch(pol,[ids[i].tolist()],[[2,3][i]])
        torch.testing.assert_close(got['sums'][i],one['sums'][0],atol=1e-5,rtol=1e-5)


def test_attention_sketch_ignores_left_padding():
    p=torch.tensor([[[[.1,.2,.7]]]])
    a=attention_sketch(p,torch.ones(1,3,dtype=torch.bool))
    b=attention_sketch(F.pad(p,(2,0)),torch.tensor([[0,0,1,1,1]],dtype=torch.bool))
    torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-6)


def test_hellaswag_standard_character_normalization_and_preprocessing():
    from scripts.mistral_hellaswag import choice_metrics
    from scripts.prepare_mistral_comparisons import process
    r=choice_metrics([-2.,-3.,-4.,-5.],['a','bbbbbbbbbb','cc','dd'],1)
    assert r['prediction']==0 and r['prediction_normalized']==1 and r['accuracy_normalized']==1
    doc={'activity_label':'Test','ctx_a':'A [title] B','ctx_b':'he Moves','endings':[' [foo] Hi  there ']*4,'label':'2'}
    p=process(doc);assert p['query']=='Test: A. B He moves' and p['choices'][0]==' Hi there' and p['gold']==2
