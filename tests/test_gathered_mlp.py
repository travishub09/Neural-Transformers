import pytest
import torch
from transformers import MistralConfig,MistralForCausalLM
from pathpred.gated_qwen import gate_mistral
from pathpred.gathered_mlp import gathered_mlp,copy_mlp_to_cpu,TransferMeter


def test_gather_matches_masked_dense_and_preserves_observations():
    torch.manual_seed(9)
    model=MistralForCausalLM(MistralConfig(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)).eval()
    ctrl=gate_mistral(model);x=torch.randn(1,1,16)
    for retained in [0,1,11,24]:
        gate=torch.zeros(1,1,24);gate[:,:,:retained]=1
        with ctrl.use_gates({(1,'mlp'):gate}):expected=model.model.layers[1].mlp(x)
        observed=[]
        hook=model.model.layers[1].mlp.down_proj.register_forward_pre_hook(lambda m,args:observed.append(args[0]))
        try:
            with gathered_mlp(model,ctrl),ctrl.use_gates({(1,'mlp'):gate}):actual=model.model.layers[1].mlp(x)
        finally:hook.remove()
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
        assert (observed[-1][:,:,retained:]==0).all()


@pytest.mark.parametrize('packed',[False,True])
def test_cpu_payload_counts_selected_weights(packed):
    model=MistralForCausalLM(MistralConfig(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)).eval()
    ctrl=gate_mistral(model);cpu=copy_mlp_to_cpu(model,packed_down=packed);meter=TransferMeter()
    gate=torch.zeros(1,1,24);gate[:,:,:7]=1
    x=torch.randn(1,1,16)
    with ctrl.use_gates({(1,'mlp'):gate}):expected=model.model.layers[1].mlp(x)
    with gathered_mlp(model,ctrl,cpu_weights=cpu,meter=meter),ctrl.use_gates({(1,'mlp'):gate}):
        actual=model.model.layers[1].mlp(x)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    assert meter.bytes==3*7*16*4 and meter.matrices==3
