"""Boundary adapter: convert a completed LLM layer's trace to discrete input."""
import torch
from .path_signals import UnitThresholds


def encode_layer_norms(thresholds, norms, observed, layer, kind='binary'):
    if not 0 <= layer < thresholds.upper.shape[0]:
        raise ValueError('Layer outside the calibrated model')
    subset = UnitThresholds(thresholds.upper[layer:layer + 1], thresholds.lower[layer:layer + 1])
    return subset.encode(norms.unsqueeze(-2), observed.unsqueeze(-2), kind).squeeze(-2)


def encode_trace_layer(thresholds, trace, layer, kind='binary'):
    """Call after BOTH attention and MLP of layer have completed.

    Returns [batch, token_positions, units]; flatten token dimensions for PathModel.
    No scores are applied here. Routing untrained predictions would be invalid.
    """
    attention = trace.records[(layer, 'attention')]
    mlp = trace.records[(layer, 'mlp')]
    norms = torch.cat((attention.observed_norms, mlp.observed_norms), dim=-1)
    observed = torch.cat((attention.selected, mlp.selected), dim=-1)
    return encode_layer_norms(thresholds, norms, observed, layer, kind)
