import copy
import hashlib
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM
from pathpred import gate_qwen3
from pathpred.clustering import reorder_mlp_
from pathpred.reordering import enable_canonical_mlp_order, load_clustered_qwen


def pair(dtype=torch.float32, device='cpu'):
    torch.manual_seed(82)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=128, attention_dropout=0., tie_word_embeddings=True)
    config._attn_implementation = 'eager'
    original = Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval().requires_grad_(False)
    model = copy.deepcopy(original)
    permutations = torch.stack([torch.randperm(64) for _ in range(3)])
    identities = {n: id(p) for n, p in model.named_parameters()}
    for layer, perm in zip(model.model.layers, permutations):
        reorder_mlp_(layer.mlp, perm)
    enable_canonical_mlp_order(model, permutations)
    assert identities == {n: id(p) for n, p in model.named_parameters()}
    assert set(model.state_dict()) == set(original.state_dict())
    return original, model, permutations


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_canonical_reordering_is_bitwise_exact_with_cache_and_all_on_gates(dtype, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    original, model, permutations = pair(dtype, device)
    ids = torch.tensor([[1, 7, 9, 5], [2, 3, 8, 4]], device=device)
    controller = gate_qwen3(model, group_size=16)
    all_on = {(l, k): torch.ones(4, device=device) for l in range(3) for k in ['mlp', 'attention']}
    with torch.inference_mode(), controller.use_gates(all_on, record=True) as trace:
        a, b = original(ids, use_cache=True), model(ids, use_cache=True)
        assert torch.equal(a.logits, b.logits)
        assert len(trace.records) == 6
        an = original(ids[:, :1], past_key_values=a.past_key_values, use_cache=True)
        bn = model(ids[:, :1], past_key_values=b.past_key_values, use_cache=True)
        assert torch.equal(an.logits, bn.logits)
    controller.remove()
    with torch.inference_mode():
        assert torch.equal(original(ids, use_cache=False).logits, model(ids, use_cache=False).logits)


def test_canonical_group_gates_and_gradients_match_original_neuron_mask():
    original, model, permutations = pair(torch.float64)
    ids = torch.tensor([[1, 7, 9, 5], [2, 3, 8, 4]])
    controller = gate_qwen3(model, group_size=16)
    gates = torch.tensor([0., .3, 1., .8], dtype=torch.float64, requires_grad=True)
    reference_gates = gates.detach().clone().requires_grad_(True)
    original_order = torch.argsort(permutations[1])
    def mask_hook(module, args):
        return (args[0] * reference_gates.repeat_interleave(16)[original_order],)
    handle = original.model.layers[1].mlp.down_proj.register_forward_pre_hook(mask_hook)
    reference = original(ids, use_cache=False).logits
    reference.square().mean().backward()
    with controller.use_gates({(1, 'mlp'): gates}, record=True, detach_observations=False):
        actual = model(ids, use_cache=False).logits
        assert torch.equal(actual, reference)
        actual.square().mean().backward()
    torch.testing.assert_close(gates.grad, reference_gates.grad, rtol=1e-12, atol=1e-12)
    assert all(p.grad is None for p in model.parameters())
    handle.remove()


def test_canonical_checkpoint_loader_and_corruption_check(tmp_path):
    original, model, permutations = pair()
    artifact = tmp_path / 'neuron-permutations.safetensors'
    save_file({'permutations': permutations}, str(artifact))
    model.config.pathpred_permutations_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    model.save_pretrained(tmp_path)
    loaded = load_clustered_qwen(tmp_path, attn_implementation='eager', local_files_only=True)
    ids = torch.tensor([[1, 9, 3]])
    with torch.inference_mode():
        assert torch.equal(original(ids).logits, loaded(ids).logits)
    assert not any(p.requires_grad for p in loaded.parameters())
    artifact.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_clustered_qwen(tmp_path, attn_implementation='eager', local_files_only=True)


def test_precision_roundtrip_preserves_rotary_and_reload(tmp_path):
    from scripts.verify_reordered_model import cast_preserving_rotary
    original, model, permutations = pair()
    initial_freq = model.model.rotary_emb.inv_freq.clone()
    cast_preserving_rotary(model, torch.bfloat16)
    cast_preserving_rotary(model, torch.float32)
    cast_preserving_rotary(model, torch.bfloat16)
    assert torch.equal(model.model.rotary_emb.inv_freq, initial_freq)
    assert model.model.rotary_emb.inv_freq.dtype == initial_freq.dtype
    artifact = tmp_path / 'neuron-permutations.safetensors'
    save_file({'permutations': permutations}, str(artifact))
    model.config.pathpred_permutations_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    model.save_pretrained(tmp_path)
    loaded = load_clustered_qwen(tmp_path, dtype=torch.bfloat16, attn_implementation='eager', local_files_only=True)
    ids = torch.tensor([[1, 9, 3, 7, 5]])
    with torch.inference_mode():
        assert torch.equal(model(ids).logits, loaded(ids).logits)
