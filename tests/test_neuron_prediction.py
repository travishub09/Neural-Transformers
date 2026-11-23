import torch
import pytest
from transformers import Qwen3Config,Qwen3ForCausalLM
from pathpred import gate_qwen3
from pathpred.path_model import PathConfig
from pathpred.neuron_prediction import (NeuronPredictor,collect_dense,route_neurons,
    stack_stream,recall_weighted_loss,recovered_advantage)


def predictor(signal='binary'):
    torch.manual_seed(8);torch.set_num_threads(2)
    return NeuronPredictor(PathConfig(llm_layers=6,units=16,width=16,depth=2,heads=4,
        hidden_size=16,hidden_summary=8,signal=signal))


def test_token_embedding_reuse_matches_step_and_is_causal():
    for signal in ['binary','hidden']:
        p=predictor(signal)
        stream=torch.randint(-1,2,(2,5,6,16),dtype=torch.int8) if signal=='binary' else torch.randn(2,5,6,16)
        key='states' if signal=='binary' else 'hidden'
        expected=p.forward_tokens(**{key:stream})
        cache=None;outputs=[]
        for l in range(4):
            r=p.step_tokens(**{key:stream[:,:,l]},cache=cache);outputs.append(r.logits);cache=r.cache
        torch.testing.assert_close(expected,torch.stack(outputs,2),rtol=2e-5,atol=2e-6)
        changed=stream.clone();changed[:,3:]=0;changed[:,:,3:]=0
        torch.testing.assert_close(expected[:,:3,:3],p.forward_tokens(**{key:changed})[:,:3,:3],rtol=0,atol=0)
        changed=stream.clone();changed[1]=0
        torch.testing.assert_close(expected[0],p.forward_tokens(**{key:changed})[0],rtol=0,atol=0)


def tiny_qwen():
    torch.manual_seed(17)
    cfg=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=16,num_hidden_layers=6,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8)
    cfg._attn_implementation='eager'
    return Qwen3ForCausalLM(cfg).eval()


def test_online_router_masks_unknowns_keeps_prefix_and_backbone_frozen():
    model=tiny_qwen();controller=gate_qwen3(model,1);p=predictor().eval()
    ids=torch.tensor([[2,4,6,8],[3,5,7,9]])
    q=torch.ones(6,16)*.001
    with torch.no_grad(),controller.use_gates(),route_neurons(model,controller,q,.5,predictor=p) as trace:
        model(ids,use_cache=False)
    for l in range(6):
        assert (trace['masks'][l].sum(-1)==(16 if l<2 else 8)).all()
        assert (trace['bits'][l][~trace['masks'][l]]==-1).all()
        assert set(trace['bits'][l].unique().tolist())<= {-1,0,1}
    states=stack_stream(trace,'bits')
    logits=p.forward_tokens(states=states)
    assert logits.shape==(2,4,4,16)
    loss=recall_weighted_loss(logits,torch.ones_like(logits,dtype=torch.bool));loss.backward()
    assert all(x.grad is None and not x.requires_grad for x in model.parameters())
    assert any(x.grad is not None for x in p.parameters())


def test_previous_token_embedding_survives_separate_decode_calls():
    p=predictor().eval();states=torch.randint(-1,2,(1,3,6,16),dtype=torch.int8)
    expected=p.forward_tokens(states=states)
    previous=None;actual=[]
    for t in range(3):
        cache=None;next_embeddings={};layers=[]
        for l in range(4):
            r=p.step_tokens(states=states[:,t:t+1,l],cache=cache,
                previous_embedding=None if previous is None else previous[l],embedding_store=next_embeddings)
            layers.append(r.logits);cache=r.cache
        actual.append(torch.stack(layers,2));previous=next_embeddings
    torch.testing.assert_close(torch.cat(actual,1),expected,rtol=2e-5,atol=2e-6)


def test_all_on_and_teacher_labels_are_separate_from_router_input():
    model=tiny_qwen();controller=gate_qwen3(model,1);ids=torch.tensor([[2,3,4]])
    q=torch.zeros(6,16)
    with torch.no_grad(),controller.use_gates(),collect_dense(model,q) as dense:
        expected=model(ids,use_cache=False).logits
    with torch.no_grad(),controller.use_gates(),route_neurons(model,controller,q,1.,method='oracle'):
        actual=model(ids,use_cache=False).logits
    assert torch.equal(expected,actual)
    for l in range(6):
        assert (dense['labels'][l]['0.5'].sum(-1)==8).all()
    assert recovered_advantage(.4,.2,.1)==(.4-.2)/(.4-.1)
    assert recovered_advantage(.1,.2,.2) is None


def test_full_sequence_and_cached_qwen_use_the_same_neuron_decisions():
    for signal in ['binary','hidden']:
        model=tiny_qwen();controller=gate_qwen3(model,1);p=predictor(signal).eval()
        ids=torch.tensor([[2,4,6,8]]);q=torch.ones(6,16)*.001
        with torch.no_grad(),controller.use_gates(),route_neurons(model,controller,q,.5,predictor=p) as trace:
            full=model(ids,use_cache=False).logits
        masks=stack_stream(trace,'masks');cache=None;previous=None;pieces=[]
        for t in range(ids.shape[1]):
            next_embeddings={}
            with torch.no_grad(),controller.use_gates(),route_neurons(model,controller,q,.5,predictor=p,
                    previous_embeddings=previous,next_embeddings=next_embeddings) as part:
                out=model(ids[:,t:t+1],past_key_values=cache,use_cache=True)
            assert torch.equal(stack_stream(part,'masks'),masks[:,t:t+1])
            pieces.append(out.logits);cache=out.past_key_values;previous=next_embeddings
        torch.testing.assert_close(torch.cat(pieces,1),full,rtol=3e-5,atol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA BF16 storage audit')
def test_bf16_matrix_storage_preserves_autocast_logits():
    for signal in ['binary','hidden']:
        p=predictor(signal).cuda().eval()
        stream=torch.randint(-1,2,(2,3,6,16),device='cuda',dtype=torch.int8) if signal=='binary' else torch.randn(2,3,6,16,device='cuda')
        kw={'states':stream} if signal=='binary' else {'hidden':stream}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            expected=p.forward_tokens(**kw)
            for name,parameter in p.named_parameters():
                if parameter.ndim>=2 and name not in ('layer_embedding','output_bias'):
                    parameter.data=parameter.data.to(torch.bfloat16)
            # Clear autocast's parameter cache before checking the changed storage.
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            actual=p.forward_tokens(**kw)
        assert torch.equal(expected,actual)
