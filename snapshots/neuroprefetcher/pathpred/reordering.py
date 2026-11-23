"""Canonical dense execution for physically clustered MLP weights.

Storage stays clustered. Temporary gathers restore the original GEMM layout,
then expose intermediate activations in clustered order to the gate controller.
This intentionally adds overhead: it is an exact dense reference, not a sparse
kernel. Native reordered GEMMs have different BF16 rounding and are experimental.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def ordered_linear(x, weight, bias, mode=None, permutation=None, inverse=None):
    if mode is None:
        return F.linear(x, weight, bias)
    if mode == 'rows':
        original_bias = None if bias is None else bias.index_select(0, inverse)
        original = F.linear(x, weight.index_select(0, inverse), original_bias)
        return original.index_select(-1, permutation)
    if mode == 'columns':
        return F.linear(x.index_select(-1, inverse), weight.index_select(1, inverse), bias)
    raise ValueError('Unknown canonical projection mode')


class CanonicalLinear(nn.Linear):
    """Weight-sharing Linear with original-order GEMMs and clustered coordinates."""
    def __init__(self, original, permutation, mode):
        nn.Module.__init__(self)
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.canonical_mode = mode
        self.register_buffer('canonical_permutation', permutation.to(original.weight.device).clone(), persistent=False)
        self.register_buffer('canonical_inverse', torch.argsort(self.canonical_permutation), persistent=False)
        self.train(original.training)

    def forward(self, x):
        return ordered_linear(x, self.weight, self.bias, self.canonical_mode,
                              self.canonical_permutation, self.canonical_inverse)


def enable_canonical_mlp_order(model, permutations):
    """Enable BEFORE gate_qwen3, after applying physical weight permutations.

    No weights are changed here. All temporary gathers are differentiable, and
    nonpersistent order buffers do not change the checkpoint's weight keys.
    """
    width = model.config.intermediate_size
    if (permutations.dtype != torch.int64 or
            permutations.shape != (model.config.num_hidden_layers, width) or
            not torch.equal(permutations.cpu().sort(dim=1).values,
                            torch.arange(width).expand(model.config.num_hidden_layers, -1))):
        raise ValueError('Invalid layer permutations')
    entries = []
    for layer, perm in zip(model.model.layers, permutations):
        for name, mode in [('gate_proj', 'rows'), ('up_proj', 'rows'), ('down_proj', 'columns')]:
            original = getattr(layer.mlp, name)
            if type(original) is not nn.Linear:
                raise TypeError('Enable canonical execution once, before installing gates')
            entries.append((layer.mlp, name, original, perm, mode))
    for parent, name, original, perm, mode in entries:
        setattr(parent, name, CanonicalLinear(original, perm, mode))
    return model


def load_clustered_qwen(checkpoint, **kwargs):
    """Load the self-contained grouped checkpoint in exact dense reference mode.

    Plain AutoModelForCausalLM loads native reordered GEMMs instead. Use this
    loader for correctness/training, and report its gather overhead in any timing.
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    directory = Path(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(directory, **kwargs).eval().requires_grad_(False)
    artifact = directory / 'neuron-permutations.safetensors'
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != getattr(model.config, 'pathpred_permutations_sha256', None):
        raise ValueError('Checkpoint permutation checksum mismatch')
    permutations = load_file(str(artifact))['permutations']
    return enable_canonical_mlp_order(model, permutations)
