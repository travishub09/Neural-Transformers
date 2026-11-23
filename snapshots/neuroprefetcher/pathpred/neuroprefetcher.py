"""Paper-specified NeuroPrefetcher predictor, adapted to Qwen's dimensions.

Reference: Dhar et al., arXiv:2608.22643, sections 3.2--3.4.
This implements the paper's per-block residual/SiLU architecture, not the
released trainer's single outer residual or runtime's GELU substitution.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn


@dataclass(frozen=True)
class NeuroConfig:
    hidden_size: int = 2048
    units: int = 6144
    layers: int = 28
    prefix: int = 2
    centroids: int = 512
    width: int = 512
    depth: int = 4
    sketch_size: int = 128
    dropout: float = .1


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8*width, 4*width), nn.LayerNorm(4*width),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(4*width, 8*width), nn.LayerNorm(8*width))

    def forward(self, x):
        return x + self.net(x)


class NeuroPredictor(nn.Module):
    def __init__(self, config=NeuroConfig()):
        super().__init__(); self.config = config; c = config
        def project(n):
            return nn.Sequential(nn.Linear(n, c.width), nn.LayerNorm(c.width),
                                 nn.SiLU(), nn.Dropout(c.dropout))
        self.projections = nn.ModuleList([project(n) for n in
            (c.hidden_size, c.hidden_size, c.sketch_size, 1)])
        self.centroid_embedding = nn.Sequential(nn.Embedding(c.centroids, 4*c.width),
            nn.Linear(4*c.width, 4*c.width), nn.LayerNorm(4*c.width), nn.SiLU())
        self.trunk = nn.Sequential(*[ResidualBlock(c.width, c.dropout) for _ in range(c.depth)])
        self.gate_heads = nn.ModuleList([nn.Linear(8*c.width, c.centroids) for _ in range(c.layers-c.prefix)])
        self.up_heads = nn.ModuleList([nn.Linear(8*c.width, c.centroids) for _ in range(c.layers-c.prefix)])

    def forward(self, features, centroid_id):
        if len(features) != 4: raise ValueError('Four continuous feature branches required')
        x = torch.cat([projection(value) for projection, value in zip(self.projections, features)] +
                      [self.centroid_embedding(centroid_id)], dim=-1)
        x = self.trunk(x)
        return torch.stack([head(x) for head in self.gate_heads], dim=-2), torch.stack([head(x) for head in self.up_heads], dim=-2)

    def accounting(self, base_parameters=None):
        parameters = sum(p.numel() for p in self.parameters())
        linear_flops = 2*sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Linear))
        result = {'config': asdict(self.config), 'parameters': parameters,
                  'bf16_weight_bytes': 2*parameters, 'linear_flops_per_token': linear_flops,
                  'actual_parameter_storage_bytes': sum(p.numel()*p.element_size() for p in self.parameters()),
                  'bf16_weight_bytes_is_deployment_estimate': True,
                  'excluded_overhead': ['centroid matching and lookup', 'attention sketch',
                      'normalization/nonlinearities', 'kernel launch', 'centroid bank storage'],
                  'sparse_speedup_measured': False}
        if base_parameters is not None: result['parameter_share_of_qwen'] = parameters/base_parameters
        return result


def attention_sketch(probabilities, *, cached_positions=None, size=128):
    """Four statistics/head as in reference runtime, zero-pad Qwen16 heads.

    probabilities: [batch, heads, query, key]. Padding to128 (16*4=64 actual
    values) preserves the paper's input size without inventing extra heads.
    Normalize argmax by each query's causal context, never future sequence length.
    """
    p = probabilities.float(); b, h, q, k = p.shape
    if cached_positions is None:
        cached_positions = torch.arange(k-q, k, device=p.device)
    denominator = torch.as_tensor(cached_positions, device=p.device).clamp_min(1)
    maximum, arg = p.max(-1)
    entropy = -(p*p.clamp_min(1e-30).log()).sum(-1)
    top = p.topk(min(4, k), dim=-1).values.sum(-1)
    x = torch.stack([maximum, arg.float()/denominator, entropy, top], dim=-1)
    x = x.transpose(1, 2).reshape(b, q, 4*h)
    if x.shape[-1] > size: raise ValueError('Sketch cannot truncate heads silently')
    return torch.nn.functional.pad(x, (0, size-x.shape[-1]))


def prune_aggregate(aggregate, keep):
    """Deterministic exact-cardinality centroid; ties resolve by neuron index."""
    count = int(round(aggregate.shape[-1]*keep))
    if not 0 <= count <= aggregate.shape[-1]: raise ValueError('Invalid retention')
    order = torch.argsort(aggregate, dim=-1, descending=True, stable=True)
    result = torch.zeros_like(aggregate, dtype=torch.bool)
    return result.scatter_(-1, order[..., :count], True)


def nearest_centroid(binary, centroids):
    """Hamming distance, including unequal centroid sizes; first ID wins ties."""
    x = binary.float(); c = centroids.float()
    distance = x.sum(-1, keepdim=True) + c.sum(-1) - 2*x@c.transpose(-1, -2)
    return distance.argmin(-1)


def fit_centroids(binary, *, k=512, keep=.5, iterations=20, seed=20261113, batch_size=2048):
    """Binary Lloyd clustering with aggregate-and-prune updates, bounded memory.

    One bank per layer/projection, fit exclusively on training samples.
    Explicitly report empty/duplicate clusters; never conceal vocabulary collapse.
    """
    if binary.ndim != 2 or len(binary) < k: raise ValueError('Need at least K training masks')
    rng = torch.Generator(device=binary.device).manual_seed(seed)
    indices = torch.randperm(len(binary), generator=rng, device=binary.device)[:k]
    aggregate = binary[indices].float(); centroids = prune_aggregate(aggregate, keep)
    last = None; history = []
    for iteration in range(iterations):
        assignments = torch.cat([nearest_centroid(x, centroids) for x in binary.split(batch_size)])
        sums = torch.zeros_like(aggregate); counts = torch.bincount(assignments, minlength=k)
        for start in range(0, len(binary), batch_size):
            sums.index_add_(0, assignments[start:start+batch_size], binary[start:start+batch_size].float())
        occupied = counts > 0
        aggregate[occupied] = sums[occupied]/counts[occupied,None]
        centroids = prune_aggregate(aggregate, keep)
        history.append({'iteration': iteration, 'empty_clusters': int((~occupied).sum()),
                        'changed_assignments': None if last is None else int((assignments != last).sum())})
        if last is not None and torch.equal(assignments, last): break
        last = assignments
    return {'aggregate': aggregate, 'masks': centroids, 'history': history,
            'unique_centroids': len(torch.unique(centroids, dim=0)), 'counts': counts}


def reused_fraction(current, previous):
    """Fraction of current active rows already active for previous token.

    Returns numerator/denominator separately; zero active sets are undefined,
    and callers exclude first tokens and never cross document boundaries.
    """
    return (current & previous).sum(-1), current.sum(-1)


def qwen_overhead(predictor,model,context=1024):
    c=model.config;h=c.hidden_size;q=c.num_attention_heads*c.head_dim;kv=c.num_key_value_heads*c.head_dim
    per_layer=6*h*c.intermediate_size+4*h*(q+kv)+4*context*q
    dense=per_layer*c.num_hidden_layers+2*h*c.vocab_size
    result=predictor.accounting(sum(v.numel() for v in model.parameters()))
    matching=2*predictor.config.centroids*c.intermediate_size
    result.update(reference_decode_context=context,dense_qwen_matmul_flops=dense,
        prefix_centroid_matching_matmul_flops=matching,
        predictor_plus_matching_flops_share=(result['linear_flops_per_token']+matching)/dense,
        runtime_centroid_bank_bool_bytes=(2*(c.num_hidden_layers-2)+1)*predictor.config.centroids*c.intermediate_size,
        centroid_storage_note='One byte per bool in this implementation; packed-bit lower bound is 1/8. Training aggregate banks excluded from runtime footprint.',
        dense_qwen_bf16_weight_bytes=2*sum(v.numel() for v in model.parameters()))
    result['actual_qwen_parameter_storage_bytes']=sum(v.numel()*v.element_size() for v in model.parameters())
    result['actual_predictor_to_qwen_storage_ratio']=result['actual_parameter_storage_bytes']/result['actual_qwen_parameter_storage_bytes']
    result['measurement_note']='Prototype predictor parameters remain float32 under bf16 autocast; conversion/cache overhead must be measured, not equated with bf16 deployment bytes.'
    return result
