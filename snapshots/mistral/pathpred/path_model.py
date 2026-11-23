"""Causal transformer over LLM layers, using binary observed activity by default.

One batch row is one LLM token. Sequence positions are completed LLM layers.
An observation at l predicts layer l+lookahead. The first lookahead LLM layers
are explicitly dense. An incremental cache is reset for each new LLM token;
previous-token inputs contain only the same layer, never its later layers.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PathConfig:
    llm_layers: int = 28
    units: int = 64
    width: int = 128
    depth: int = 2
    heads: int = 4
    expansion: int = 4
    lookahead: int = 2
    previous_token: bool = True
    signal: str = 'binary'
    hidden_size: int = 2048
    hidden_summary: int = 64

    def __post_init__(self):
        if (min(self.llm_layers, self.units, self.width, self.depth, self.heads, self.expansion) < 1
                or self.width % self.heads or not 1 <= self.lookahead < self.llm_layers):
            raise ValueError('Invalid path model dimensions')
        if self.signal not in ('binary', 'decisions', 'coarse', 'magnitudes', 'hidden', 'hybrid', 'combined'):
            raise ValueError('Unknown path signal')
        if self.signal in ('hidden', 'hybrid') and min(self.hidden_size, self.hidden_summary) < 1:
            raise ValueError('Invalid hidden-state dimensions')
        if self.signal == 'combined' and self.hidden_size < 1:
            raise ValueError('Invalid hidden-state dimensions')


@dataclass(frozen=True)
class PathCache:
    next_layer: int
    blocks: tuple  # one (keys, values) pair per transformer block
    batch_size: int
    owner: int


@dataclass(frozen=True)
class LayerPrediction:
    target_layer: int
    logits: torch.Tensor
    cache: PathCache


class CausalBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.width // cfg.heads
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.out = nn.Linear(cfg.width, cfg.width)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, cfg.expansion * cfg.width), nn.GELU(),
                                 nn.Linear(cfg.expansion * cfg.width, cfg.width))

    def forward(self, x, cache=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        # Cached calls contain exactly ONE new layer and all keys are in its past.
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=cache is None)
        x = x + self.out(attended.transpose(1, 2).reshape(batch, length, width))
        x = x + self.mlp(self.norm2(x))
        return x, (k, v)


class PathModel(nn.Module):
    def __init__(self, config=PathConfig()):
        super().__init__()
        self.config = config
        c = config
        self.steps = c.llm_layers - c.lookahead
        self.layer_embedding = nn.Parameter(torch.empty(self.steps, c.width))
        # Equal feature-slot width for the discrete and magnitude comparisons.
        if c.signal != 'hidden':
            self.activity_projection = nn.Parameter(torch.empty(self.steps, 2 * c.units, c.width))
        else:
            self.register_parameter('activity_projection', None)
        if c.signal in ('hidden', 'hybrid'):
            self.hidden_compression = nn.Parameter(torch.empty(self.steps, c.hidden_size, c.hidden_summary))
            self.hidden_projection = nn.Parameter(torch.empty(self.steps, c.hidden_summary, c.width))
        else:
            self.register_parameter('hidden_compression', None)
            self.register_parameter('hidden_projection', None)
        self.previous_projection = nn.Linear(c.width, c.width, bias=False) if c.previous_token else None
        self.previous_available = nn.Parameter(torch.zeros(c.width)) if c.previous_token else None
        self.blocks = nn.ModuleList([CausalBlock(c) for _ in range(c.depth)])
        self.final_norm = nn.LayerNorm(c.width)
        self.output_weight = nn.Parameter(torch.empty(self.steps, c.width, c.units))
        self.output_bias = nn.Parameter(torch.zeros(self.steps, c.units))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.layer_embedding, std=.02)
        for p in (self.activity_projection, self.hidden_compression, self.hidden_projection, self.output_weight):
            if p is not None:
                nn.init.normal_(p, std=1 / math.sqrt(p.shape[-2]))

    @property
    def target_layers(self):
        return tuple(range(self.config.lookahead, self.config.llm_layers))

    def _features(self, states):
        c = self.config
        if c.signal in ('binary', 'hybrid', 'combined', 'decisions', 'coarse'):
            if states.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise TypeError('Discrete path observations must be integer codes, never magnitudes')
            maximum = 2 if c.signal == 'coarse' else 1
            if bool(((states < -1) | (states > maximum)).any()):
                raise ValueError('Invalid discrete observation code')
        elif c.signal == 'magnitudes':
            if not states.is_floating_point() or not torch.isfinite(states).all() or ((states < 0) & (states != -1)).any():
                raise ValueError('Magnitude ablation requires finite nonnegative values or -1')
        observed = states >= 0
        value = torch.where(observed, states, 0).to(self.layer_embedding.dtype)
        observed = observed.to(value.dtype)
        if c.signal == 'decisions':
            return torch.cat((observed, torch.zeros_like(observed)), dim=-1)
        if c.signal == 'coarse':
            value = value / 2
        return torch.cat((value, observed), dim=-1)

    def _embed(self, states, hidden, layers):
        if self.config.signal == 'hidden':
            if states is not None:
                raise ValueError('Hidden-only comparison must not receive activation observations')
            if hidden is None:
                raise ValueError('Hidden states required')
            result = None
        else:
            if states is None or states.ndim != 3 or states.shape[-1] != self.config.units:
                raise ValueError('Expected observations [tokens, completed_layers, units]')
            result = torch.einsum('nsu,sud->nsd', self._features(states), self.activity_projection[layers])
        if self.hidden_compression is not None:
            if hidden is None or hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size or not torch.isfinite(hidden).all():
                raise ValueError('Expected finite residual states [tokens, completed_layers, hidden_size]')
            h = hidden.to(self.layer_embedding.dtype)
            h = torch.einsum('nsh,shk->nsk', h, self.hidden_compression[layers])
            h = torch.einsum('nsk,skd->nsd', h, self.hidden_projection[layers])
            result = h if result is None else result + h
        elif hidden is not None:
            raise ValueError('Hidden-state input is forbidden for this signal')
        return result

    def _inputs(self, states, previous, hidden, previous_hidden, previous_valid, layers):
        x = self._embed(states, hidden, layers)
        if self.config.previous_token:
            if previous is None and previous_hidden is None:
                if previous_valid is not None and bool(previous_valid.any()):
                    raise ValueError('Previous availability declared without previous observations')
            else:
                if previous_valid is None or previous_valid.dtype != torch.bool or previous_valid.shape != x.shape[:2]:
                    raise ValueError('Previous token requires an explicit per-layer availability mask')
                # Missing previous tokens are sanitized BEFORE projection, even for hidden baselines.
                previous = None if previous is None else torch.where(previous_valid[..., None], previous, -1)
                previous_hidden = None if previous_hidden is None else torch.where(previous_valid[..., None], previous_hidden, 0.)
                px = self._embed(previous, previous_hidden, layers)
                x = x + torch.where(previous_valid[..., None], self.previous_projection(px) + self.previous_available, 0.)
        elif any(v is not None for v in (previous, previous_hidden, previous_valid)):
            raise ValueError('Previous-token inputs disabled by this configuration')
        return x + self.layer_embedding[layers]

    def forward(self, states=None, *, previous=None, hidden=None, previous_hidden=None, previous_valid=None):
        """Returns [tokens, L-lookahead, units], predicting target_layers.

        Inputs can include all L layers or just the first L-lookahead. Extra last
        layers are ignored: they cannot affect any score. No text/token IDs enter.
        """
        source = hidden if self.config.signal == 'hidden' else states
        if source is None or source.ndim != 3 or source.shape[1] not in (self.steps, self.config.llm_layers):
            raise ValueError('Expected a full layer sequence')
        size = source.shape[:2]
        for value in (states, previous, hidden, previous_hidden):
            if value is not None and value.shape[:2] != size:
                raise ValueError('All streams must align by token and layer')
        if previous_valid is not None and previous_valid.shape != size:
            raise ValueError('Previous availability must align with layers')
        def prefix(x):
            return None if x is None else x[:, :self.steps]
        layers = slice(0, self.steps)
        x = self._inputs(prefix(states), prefix(previous), prefix(hidden), prefix(previous_hidden), prefix(previous_valid), layers)
        for block in self.blocks:
            x, _ = block(x)
        return torch.einsum('nsd,sdu->nsu', self.final_norm(x), self.output_weight) + self.output_bias

    def step(self, states=None, *, cache=None, previous=None, hidden=None, previous_hidden=None, previous_valid=None):
        """Append one completed layer; callers must reset cache for each LLM token."""
        source = hidden if self.config.signal == 'hidden' else states
        if source is None or source.ndim != 2:
            raise ValueError('Step inputs must be [tokens, features]')
        layer = 0 if cache is None else cache.next_layer
        if layer >= self.steps:
            raise ValueError('No future target remains; reset cache for a new token')
        if cache is not None and (cache.owner != id(self) or cache.batch_size != source.shape[0] or len(cache.blocks) != len(self.blocks)):
            raise ValueError('Cache belongs to a different token batch or model')
        for value in (states, previous, hidden, previous_hidden):
            if value is not None and (value.ndim != 2 or value.shape[0] != source.shape[0]):
                raise ValueError('Step streams must align')
        if previous_valid is not None and previous_valid.shape != source.shape[:1]:
            raise ValueError('Previous validity must be per token')
        def expand(x):
            return None if x is None else x[:, None]
        x = self._inputs(expand(states), expand(previous), expand(hidden), expand(previous_hidden),
                         expand(previous_valid), slice(layer, layer + 1))
        updated = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if cache is None else cache.blocks[i])
            updated.append(kv)
        logits = self.final_norm(x[:, 0]) @ self.output_weight[layer] + self.output_bias[layer]
        return LayerPrediction(layer + self.config.lookahead, logits,
                               PathCache(layer + 1, tuple(updated), source.shape[0], id(self)))

    def initial_dense_mask(self, batch_size, *, device=None):
        return torch.ones(batch_size, self.config.lookahead, self.config.units, dtype=torch.bool,
                          device=self.layer_embedding.device if device is None else device)

    def accounting(self):
        c = self.config
        steps, d = self.steps, c.width
        # Multiply-add=2 FLOPs; dense projection-only estimate, excludes pointwise ops.
        attention = sum(4 * (i + 1) * d for i in range(steps)) * c.depth
        body = steps * c.depth * (8 * d * d + 4 * c.expansion * d * d)
        input_cost = 0 if c.signal == 'hidden' else steps * 4 * c.units * d
        if self.hidden_compression is not None:
            input_cost += steps * (2 * c.hidden_size * c.hidden_summary + 2 * c.hidden_summary * d)
        previous = input_cost + steps * 2 * d * d if c.previous_token else 0
        output = steps * 2 * d * c.units
        return {'config': asdict(c), 'parameters': sum(p.numel() for p in self.parameters()),
                'matmul_flops_per_llm_token': body + attention + input_cost + previous + output,
                'cache_elements_per_llm_token': c.depth * 2 * steps * d,
                'assumptions': ['One complete incremental traversal per LLM token, including previous-token projections',
                                'Multiply-add=2 FLOPs; excludes norms, GELU, comparisons, indexing and launch overhead',
                                'No wall-clock or sparse-inference speedup implied']}
