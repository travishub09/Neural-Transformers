"""Evaluate arbitrary fresh neuron groups without changing model weights."""
import torch
from .clustering import validate_permutation
from .sparsity_diagnostics import block_gram, scores, select_mask


def original_to_stored_order(fresh_original, stored_original):
    """Both permutations list original neuron IDs; return stored column indices."""
    validate_permutation(fresh_original, len(stored_original))
    validate_permutation(stored_original, len(stored_original))
    return torch.argsort(stored_original)[fresh_original]


def contribution_mask(x, permutation, group_size, keep, gram):
    """Choose groups in permuted coordinates, return mask in model coordinates."""
    if x.shape[-1] % group_size:
        raise ValueError('Group size must divide neuron count')
    if keep == 1:
        return torch.ones_like(x, dtype=torch.bool)
    grouped_x = x.index_select(-1, permutation)
    values = scores(grouped_x, group_size, 'contribution', None, gram)
    selected = select_mask(values, keep).repeat_interleave(group_size, dim=-1)
    return torch.zeros_like(selected).scatter_(-1, permutation.expand_as(selected), selected)


def contribution_gram(weight, permutation, group_size):
    validate_permutation(permutation, weight.shape[-1])
    return block_gram(weight.index_select(1, permutation), group_size)
