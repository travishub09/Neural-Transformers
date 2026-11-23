"""Phase 0 targets, explicit cost budgets, censored inputs, and ranking metrics."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def coverage_labels(norms, coverage=.95, families=((0, 16), (16, 64))):
    """Smallest top-norm prefix reaching coverage, independently per family.

    Stable index order breaks ties. All-zero families have no positive targets.
    Returns labels and family-balanced norm shares for secondary diagnostics.
    """
    if norms.ndim < 2 or not 0 < coverage <= 1 or not torch.isfinite(norms).all() or (norms < 0).any():
        raise ValueError('Invalid norm targets')
    if [i for a, b in families for i in range(a, b)] != list(range(norms.shape[-1])):
        raise ValueError('Families must partition the unit dimension in order')
    labels = torch.zeros_like(norms, dtype=torch.bool)
    shares = torch.zeros_like(norms, dtype=torch.float32)
    for a, b in families:
        x = norms[..., a:b].float()
        order = torch.argsort(x, dim=-1, descending=True, stable=True)
        sorted_x = x.gather(-1, order)
        total = sorted_x.sum(-1, keepdim=True)
        prior = sorted_x.cumsum(-1) - sorted_x
        selected = (prior < coverage * total) & (total > 0)
        labels[..., a:b].scatter_(-1, order, selected)
        shares[..., a:b] = x / total.clamp_min(1e-30) / len(families)
    return labels, shares


def budget_plan(costs, budget, router_flops, reference_router_flops, lookahead=2, context=1024):
    """Same total theoretical matmul cap, including router and dense warmup.

    This experiment's 1K query heads and MLP groups have equal per-unit FLOPs.
    Reject unequal costs instead of silently treating unit count as compute.
    """
    scenario = costs['scenarios'][str(context)]
    per_unit = scenario['per_layer_unit_flops']
    if len(set(per_unit)) != 1:
        raise ValueError('This fixed-count allocator requires equal per-unit FLOPs')
    if not 0 < budget <= 1 or router_flops < 0 or reference_router_flops < 0:
        raise ValueError('Invalid budget')
    layers, units = costs['layers'], len(per_unit)
    steps = layers - lookahead
    mandatory = scenario['mandatory_kv_projection_flops'] + scenario['mandatory_lm_head_flops']
    warmup = lookahead * units * per_unit[0]
    cap = mandatory + budget * scenario['all_layers_gated_flops'] + reference_router_flops
    allowance = math.floor((cap - mandatory - warmup - router_flops) / per_unit[0] + 1e-9)
    if not 0 <= allowance <= steps * units:
        raise ValueError('Budget infeasible after mandatory computation and router')
    low, remainder = divmod(allowance, steps)
    counts = [low + ((i + 1) * remainder // steps - i * remainder // steps) for i in range(steps)]
    used = mandatory + warmup + router_flops + sum(counts) * per_unit[0]
    return {'anchor': budget, 'counts': counts, 'selected_predictable_units': allowance,
            'total_matmul_cap': cap, 'accounted_matmul_used': used,
            'unused_matmul_allowance': cap - used, 'router_matmul_flops': router_flops,
            'dense_warmup_matmul_flops': warmup, 'mandatory_matmul_flops': mandatory,
            'fraction_of_dense_accounted_matmuls': used / scenario['dense_accounted_flops']}


def mask_for_budget(n, layers, units, counts, seed, device='cpu', lookahead=2):
    """Independent random masks with exact per-layer counts; warmup fully observed."""
    if len(counts) != layers - lookahead or any(not 0 <= k <= units for k in counts):
        raise ValueError('Invalid observation counts')
    gen = torch.Generator(device=device).manual_seed(int(seed))
    ranks = torch.rand((n, layers - lookahead, units), generator=gen, device=device).argsort(-1).argsort(-1)
    later = ranks < torch.tensor(counts, device=device)[None, :, None]
    return torch.cat((torch.ones(n, lookahead, units, device=device, dtype=torch.bool), later), dim=1)


def select_units(scores, counts):
    if scores.ndim != 3 or len(counts) != scores.shape[1]:
        raise ValueError('Expected [positions,predicted_layers,units] scores')
    if any(not 0 <= k <= scores.shape[-1] for k in counts) or not torch.isfinite(scores).all():
        raise ValueError('Invalid scores/counts')
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    chosen_order = torch.arange(scores.shape[-1], device=scores.device)[None, None, :] < torch.tensor(counts, device=scores.device)[None, :, None]
    chosen = torch.zeros_like(scores, dtype=torch.bool)
    return chosen.scatter_(-1, order, chosen_order.expand_as(scores))


def weighted_loss(logits, labels, positive_weight=2., families=((0, 16), (16, 64))):
    losses = F.binary_cross_entropy_with_logits(logits.float(), labels.float(),
        pos_weight=torch.tensor(positive_weight, device=logits.device), reduction='none')
    return torch.stack([losses[..., a:b].mean() for a, b in families]).mean()


def metric_rows(chosen, labels, shares, strong, families=((0, 16), (16, 64))):
    """Per-example sufficient statistics; retain document clusters for later CIs."""
    chosen, labels, strong = chosen.bool(), labels.bool(), strong.bool()
    rows = {'true_positive': (chosen & labels).sum((1, 2)), 'positive': labels.sum((1, 2)),
            'selected': chosen.sum((1, 2)), 'norm_mass': (chosen * shares).sum((1, 2)),
            'layers': torch.full((len(chosen),), chosen.shape[1], device=chosen.device),
            'strong_true_positive': (chosen & strong).sum((1, 2)), 'strong_positive': strong.sum((1, 2))}
    for name, (a, b) in zip(('heads', 'mlp'), families):
        rows[name + '_true_positive'] = (chosen[..., a:b] & labels[..., a:b]).sum((1, 2))
        rows[name + '_positive'] = labels[..., a:b].sum((1, 2))
    return {k: v.double() for k, v in rows.items()}


def summarize_rows(rows):
    sums = {k: float(np.asarray(v).sum()) for k, v in rows.items()}
    def ratio(a, b):
        return sums[a] / sums[b] if sums[b] else None
    return {'recall': ratio('true_positive', 'positive'), 'precision': ratio('true_positive', 'selected'),
            'norm_mass_fraction': ratio('norm_mass', 'layers'),
            'strong_bit_recall': ratio('strong_true_positive', 'strong_positive'),
            'head_recall': ratio('heads_true_positive', 'heads_positive'),
            'mlp_recall': ratio('mlp_true_positive', 'mlp_positive'), 'totals': sums}


def paired_document_bootstrap(a, b, positives, documents, domains, *, repetitions=2000, seed=94001):
    """Paired recall difference, resampling seeds and documents within domains.

    a/b: [seeds,examples] true positives, already averaged over fixed test masks.
    Returns an interval for data/seed variation, not for proxy-label validity.
    """
    a, b, positives = np.asarray(a), np.asarray(b), np.asarray(positives)
    docs, domains = np.asarray(documents), np.asarray(domains)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != len(docs) or positives.shape != (len(docs),):
        raise ValueError('Invalid bootstrap arrays')
    unique = np.unique(docs)
    doc_domain = []
    da, db, den = [], [], []
    for doc in unique:
        mask = docs == doc
        domain = np.unique(domains[mask])
        if len(domain) != 1:
            raise ValueError('A document spans domains')
        doc_domain.append(domain[0]); da.append(a[:, mask].sum(1)); db.append(b[:, mask].sum(1)); den.append(positives[mask].sum())
    da, db, den, doc_domain = np.stack(da, 1), np.stack(db, 1), np.array(den), np.array(doc_domain)
    strata = [np.where(doc_domain == x)[0] for x in np.unique(doc_domain)]
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        si = rng.integers(0, a.shape[0], a.shape[0])
        di = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        denominator = den[di].sum()
        samples.append((da[si][:, di].mean(0).sum() - db[si][:, di].mean(0).sum()) / denominator)
    return {'difference': float((a.mean(0).sum() - b.mean(0).sum()) / positives.sum()),
            'ci95': np.quantile(samples, [.025, .975]).tolist(), 'documents': len(unique),
            'seed_count': a.shape[0], 'bootstrap_repetitions': repetitions,
            'method': 'paired seed and document-cluster bootstrap, stratified by domain'}


def normalized_skill_bootstrap(hits, lower, upper, documents, domains, *, repetitions=2000, seed=94001):
    """(method-reference)/(oracle-reference), paired across seeds/documents.

    Inputs are per-example hit counts, so common recall denominators cancel.
    Undefined/saturated denominators are reported explicitly, never clipped.
    """
    hits, lower, upper = np.asarray(hits), np.asarray(lower), np.asarray(upper)
    docs, domains = np.asarray(documents), np.asarray(domains)
    if hits.ndim != 2 or hits.shape[1] != len(docs) or lower.shape != (len(docs),) or upper.shape != lower.shape:
        raise ValueError('Invalid normalization arrays')
    denominator = float((upper-lower).sum())
    if denominator <= 1e-12:
        return {'value': None, 'ci95': None, 'denominator_hits': denominator, 'status': 'undefined_oracle_reference_gap'}
    unique = np.unique(docs)
    numerators, gaps, domain_ids = [], [], []
    for doc in unique:
        take = docs == doc
        domain = np.unique(domains[take])
        if len(domain) != 1:
            raise ValueError('A document spans domains')
        numerators.append(hits[:,take].sum(1)-lower[take].sum())
        gaps.append((upper[take]-lower[take]).sum());domain_ids.append(domain[0])
    numerators, gaps, domain_ids = np.stack(numerators,1), np.array(gaps), np.array(domain_ids)
    strata = [np.where(domain_ids == d)[0] for d in np.unique(domain_ids)]
    rng = np.random.default_rng(seed);values=[];invalid=0
    for _ in range(repetitions):
        si=rng.integers(0,len(hits),len(hits))
        di=np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])
        gap=gaps[di].sum()
        if gap <= 1e-12:
            invalid+=1;continue
        values.append(numerators[si][:,di].mean(0).sum()/gap)
    return {'value':float((hits.mean(0).sum()-lower.sum())/denominator),
        'ci95':np.quantile(values,[.025,.975]).tolist() if values else None,
        'denominator_hits':denominator,'status':'defined' if not invalid else 'some_bootstrap_gaps_undefined',
        'undefined_bootstrap_draws':invalid,'bootstrap_repetitions':repetitions,
        'method':'paired seed/document bootstrap within domain strata; ratio recomputed each draw'}
