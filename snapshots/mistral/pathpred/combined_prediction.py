"""Add current-layer content to the trained binary path predictor.

Hidden inputs use the *same* post-decoder-layer, detached, RMS-normalized BF16
vectors as the hidden-only baseline. Normalization lives in the shared routing
hook, not in a second normalization here. Only the path embedding is reused for
the previous token; no previous-token hidden feature is introduced.
"""
from contextlib import contextmanager
from dataclasses import replace
import torch
from torch import nn
from .neuron_prediction import NeuronPredictor, neuron_config


class CombinedNeuronPredictor(NeuronPredictor):
    def __init__(self, config=None):
        super().__init__(replace(config or neuron_config(), signal='combined'))
        self.hidden_input_weight = nn.Parameter(torch.zeros(
            self.steps, self.config.hidden_size, self.config.width))
        self.reliance_mode = 'none'

    def load_path_state(self, state):
        """Strict weights-only warm start; the sole new parameter remains zero."""
        expected = set(self.state_dict()) - {'hidden_input_weight'}
        if set(state) != expected:
            raise ValueError('Path warm start has missing or unexpected parameter names')
        result = self.load_state_dict(state, strict=False)
        assert result.missing_keys == ['hidden_input_weight'] and not result.unexpected_keys
        with torch.no_grad():
            self.hidden_input_weight.zero_()

    def _history_embed(self, states, hidden, layers):
        if self.reliance_mode == 'zero_bits':
            # Preserve which units were observed, while removing their firing bits.
            states = torch.where(states >= 0, 0, states)
        return self._embed(states, None, layers)

    def _add_current_hidden(self, x, hidden, layers):
        if hidden is None or hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError('Combined predictor requires normalized post-layer hidden states')
        if not torch.isfinite(hidden).all():
            raise ValueError('Hidden states must be finite')
        if self.reliance_mode == 'no_path':
            # Remove current path, previous path, and observation-availability inputs.
            x = torch.zeros_like(x)
        if self.reliance_mode == 'no_hidden':
            hidden = torch.zeros_like(hidden)
        return x + torch.einsum('nsh,shd->nsd', hidden.to(x.dtype), self.hidden_input_weight[layers])

    @contextmanager
    def reliance(self, mode):
        if mode not in ('none', 'no_hidden', 'zero_bits', 'no_path'):
            raise ValueError(mode)
        original = self.reliance_mode
        self.reliance_mode = mode
        try:
            yield self
        finally:
            self.reliance_mode = original

    def accounting(self):
        result = super().accounting()
        extra = self.hidden_input_weight.numel()
        result['new_hidden_parameters'] = extra
        result['matmul_flops_per_llm_token'] += 2 * extra
        result['bf16_weight_payload_bytes'] = 2 * result['parameters']
        result['assumptions'].append('Current-token hidden projection only; RMS normalization in shared routing hook')
        return result


def make_predictor(signal):
    return CombinedNeuronPredictor() if signal == 'combined' else NeuronPredictor(neuron_config(signal))
