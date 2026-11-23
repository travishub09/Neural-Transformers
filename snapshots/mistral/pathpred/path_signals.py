"""One-bit observations, with missingness distinct from observed inactivity.

Calibration magnitudes never enter the main path predictor. Threshold decisions
are nondifferentiable observations; gradients still reach the predictor through
its future gate scores during later end-to-end training.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class UnitThresholds:
    upper: torch.Tensor  # [layers, units], empirical 75th percentile
    lower: torch.Tensor  # empirical 25th percentile, coarse ablation only

    @classmethod
    @torch.no_grad()
    def fit(cls, norms, train_mask):
        if norms.ndim != 3 or train_mask.shape != norms.shape[:1] or train_mask.dtype != torch.bool:
            raise ValueError('Expected [positions,layers,units] and a boolean training-position mask')
        selected = norms[train_mask].detach().float()
        if not len(selected) or not torch.isfinite(selected).all() or (selected < 0).any():
            raise ValueError('Training norms must be nonempty, finite and nonnegative')
        quartiles = torch.quantile(selected, torch.tensor([.25, .75], device=selected.device),
                                  dim=0, interpolation='linear')
        return cls(upper=quartiles[1], lower=quartiles[0])

    @torch.no_grad()
    def encode(self, norms, observed, kind='binary'):
        if norms.shape != observed.shape or observed.dtype != torch.bool or norms.shape[-2:] != self.upper.shape:
            raise ValueError('Norms/observations must match the calibrated layer/unit shape')
        visible = norms[observed]
        if not torch.isfinite(visible).all() or (visible < 0).any():
            raise ValueError('Observed norms must be finite and nonnegative')
        # Sanitize FIRST. Hidden values, including NaNs, cannot enter any feature.
        safe = torch.where(observed, norms.detach().float(), 0.)
        upper = self.upper.to(device=safe.device)
        lower = self.lower.to(device=safe.device)
        if kind == 'binary':
            values = (safe > upper).to(torch.int8)  # strict >; ties are not strong
        elif kind == 'coarse':
            # Observed low/off, weak and strong; missing is a fourth, separate state.
            values = (safe > lower).to(torch.int8) + (safe > upper).to(torch.int8)
        elif kind == 'decisions':
            values = torch.zeros_like(safe, dtype=torch.int8)
        elif kind == 'magnitudes':
            # An invertible rescaling for positive thresholds, not discretization.
            scale = torch.where(upper > 0, upper, 1.)
            values = safe / scale
        else:
            raise ValueError('Unknown observation kind')
        return torch.where(observed, values, values.new_full((), -1))


def censor_observations(states, keep):
    """Apply Phase 0 simulated skipping before the model, including previous-token data."""
    if keep.dtype != torch.bool or keep.shape != states.shape:
        raise ValueError('Keep mask must be boolean and match observations')
    return torch.where(keep, states, states.new_full((), -1))


def preceding_sample_indices(sequence_ids, positions):
    """Previous sampled position only when it is the actual preceding token.

    Never bridge separate windows/documents or gaps in the sampled positions.
    Requires sorted unique (sequence ID, token position) pairs.
    """
    if sequence_ids.ndim != 1 or sequence_ids.shape != positions.shape:
        raise ValueError('Expected matching one-dimensional sample indices')
    result = torch.full_like(sequence_ids, -1)
    if len(result) < 2:
        return result
    ordered = (sequence_ids[1:] > sequence_ids[:-1]) | ((sequence_ids[1:] == sequence_ids[:-1]) & (positions[1:] > positions[:-1]))
    if not bool(ordered.all()):
        raise ValueError('Sample keys must be sorted and unique')
    adjacent = (sequence_ids[1:] == sequence_ids[:-1]) & (positions[1:] == positions[:-1] + 1)
    result[1:] = torch.where(adjacent, torch.arange(len(result) - 1, device=result.device), -1)
    return result


def previous_observations(states, previous_indices):
    """Gather a previous token's same-layer observations; return -1 if unavailable."""
    if previous_indices.shape != states.shape[:1] or previous_indices.dtype != torch.long:
        raise ValueError('Invalid previous-token indices')
    if ((previous_indices < -1) | (previous_indices >= len(states))).any():
        raise ValueError('Previous-token index outside the sample set')
    gathered = states[previous_indices.clamp_min(0)]
    valid = previous_indices >= 0
    return torch.where(valid.reshape(-1, *([1] * (states.ndim - 1))), gathered, states.new_full((), -1))
