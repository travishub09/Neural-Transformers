"""Output-quality diagnostics, not sparse kernels or trained routing policies."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Condition:
    component: str
    group_size: int
    rule: str
    keep: float
    protection: str = 'none'

    @property
    def name(self):
        return f'{self.component}-g{self.group_size}-{self.rule}-k{round(100*self.keep)}-{self.protection}'


def projection(model, layer, kind):
    block = model.model.layers[layer]
    return block.mlp.down_proj if kind == 'mlp' else block.self_attn.o_proj


def block_gram(weight, width):
    # weight[output,input], contiguous input blocks are the gated units.
    blocks = weight.float().T.reshape(-1, width, weight.shape[0])
    return blocks @ blocks.transpose(-1, -2)


def scores(x, width, rule, column_norms, gram=None, threshold=None, tie=None):
    z = x.float().reshape(*x.shape[:-1], -1, width)
    if rule == 'magnitude':
        return z.square().sum(-1)
    if rule == 'percentile':
        # Same binary own-q75 definition as the failed test. Seeded fixed ties
        # avoid giving this baseline extra magnitude information.
        return (z.square().sum(-1).sqrt() > threshold).float() + tie * .5
    if rule == 'wina':
        return (z * column_norms.reshape(-1, width)).square().sum(-1)
    if rule == 'contribution':
        if width == 1:
            return (z.squeeze(-1) * column_norms).square()
        # ||W_G x_G||^2, retaining all within-group cross terms.
        return torch.einsum('...ui,uij,...uj->...u', z, gram, z).clamp_min(0)
    raise ValueError(rule)


def select_mask(values, keep, always=None):
    if not 0 < keep <= 1:
        raise ValueError('keep must be in (0,1]')
    count = math.ceil(values.shape[-1] * keep)
    ranking = values
    if always is not None:
        if len(always) > count:
            raise ValueError('Always-on set exceeds budget')
        ranking = values.clone()
        ranking[..., always] = float('inf')
    return torch.zeros_like(values, dtype=torch.bool).scatter_(-1, ranking.topk(count, dim=-1).indices, True)


class DynamicPolicy:
    """Select from actual current activations; optionally drive existing wrapper.

    Direct mode masks stock projection inputs. Wrapped mode supplies the same
    masks to GateController, which must use group_size=1. Attention/QKV/cache
    logic remains entirely in Transformers in either implementation.
    """
    def __init__(self, model, calibration=None):
        self.model = model
        self.calibration = calibration or {}
        self.norms = {}
        self.grams = {}
        self.ties = {}
        self.units = {}
        self.head_width = model.config.head_dim
        for l in range(model.config.num_hidden_layers):
            for kind in ['mlp', 'attention']:
                w = projection(model, l, kind).weight
                self.norms[l, kind] = w.float().norm(dim=0)
        self.latest_masks = {}

    def prepare(self, condition):
        for l in range(self.model.config.num_hidden_layers):
            for kind in ['mlp', 'attention']:
                width = condition.group_size if kind == 'mlp' else self.head_width
                key = (l, kind, width)
                module = projection(self.model, l, kind)
                n = module.in_features // width
                self.units[key] = n
                if key not in self.ties:
                    gen = torch.Generator(device=module.weight.device).manual_seed(19001 + l*1000 + width + (kind == 'attention'))
                    self.ties[key] = torch.rand(n, generator=gen, device=module.weight.device)
                if condition.rule == 'contribution' and width > 1 and key not in self.grams:
                    self.grams[key] = block_gram(module.weight, width)

    @contextmanager
    def apply(self, condition, controller=None, capture=False):
        self.prepare(condition)
        self.latest_masks = {}
        self.retention = {}
        handles = []
        layers = self.model.config.num_hidden_layers
        def make_hook(l, kind):
            width = condition.group_size if kind == 'mlp' else self.head_width
            key = (l, kind, width)
            disabled = condition.component not in (kind, 'both')
            protected = condition.protection in ('edges', 'both') and (l < 2 or l >= layers-2)
            def hook(module, args):
                x = args[0]
                if disabled or protected or condition.keep == 1:
                    mask = torch.ones((*x.shape[:-1], self.units[key]), dtype=torch.bool, device=x.device)
                else:
                    record = self.calibration.get(key, {})
                    ranking = scores(x, width, condition.rule, self.norms[l, kind],
                                     self.grams.get(key), record.get('q75'), self.ties[key])
                    always = record['always'] if condition.protection in ('always', 'both') else None
                    mask = select_mask(ranking, condition.keep, always)
                # Store only aggregate retention unless an audit needs full masks.
                self.retention[l, kind] = torch.stack([mask.sum(), mask.new_tensor(mask.numel(), dtype=torch.int64)])
                if capture:
                    self.latest_masks[l, kind] = mask.clone()
                expanded = mask.repeat_interleave(width, dim=-1)
                if controller is None:
                    return (x * expanded.to(x.dtype), *args[1:])
                controller.state.gates[l, kind] = expanded if kind == 'mlp' else mask
                return None
            return hook
        try:
            for l in range(layers):
                for kind in ['mlp', 'attention']:
                    handles.append(projection(self.model, l, kind).register_forward_pre_hook(make_hook(l, kind)))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def fractions(self):
        result = {}
        for kind in ['mlp', 'attention']:
            sums = torch.stack([v for (l, k), v in self.retention.items() if k == kind]).sum(0).double()
            result[kind] = float(sums[0] / sums[1])
        return result
