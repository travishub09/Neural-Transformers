"""Activity-oracle masks and full-distribution quality/concentration metrics."""
import math
import torch
from .phase0 import select_units


def oracle_mask(norms, thresholds, counts=None, *, mode='binary', seed=0, warmup=2):
    if norms.ndim != 3 or thresholds.shape != norms.shape[1:]:
        raise ValueError('Expected [positions,layers,units] norms and [layers,units] thresholds')
    if not torch.isfinite(norms).all() or (norms < 0).any() or not torch.isfinite(thresholds).all() or (thresholds < 0).any():
        raise ValueError('Norms and thresholds must be finite and nonnegative')
    strong = norms > thresholds
    if mode == 'threshold':
        selected = strong.clone()
    else:
        if counts is None:
            raise ValueError('Exact-budget mode requires counts')
        if mode in ('binary', 'random'):
            gen = torch.Generator(device=norms.device).manual_seed(seed)
            scores = torch.rand(norms.shape, generator=gen, device=norms.device) * .5
            if mode == 'binary':
                scores += strong.float()
        elif mode == 'ranked':
            scale = torch.where(thresholds > 0, thresholds, torch.ones_like(thresholds))
            # Bounded monotonic tie-break; a strong unit always outranks a weak one.
            scores = strong.float() + .5 * norms / (norms + scale)
        else:
            raise ValueError('Unknown oracle mode')
        selected = torch.ones_like(strong)
        selected[:, warmup:] = select_units(scores[:, warmup:], counts)
    selected[:, :warmup] = True
    return selected


def mask_to_gates(mask, query_heads=16):
    if mask.ndim != 3 or mask.dtype != torch.bool:
        raise ValueError('Expected boolean [positions,layers,units] masks')
    return {(layer, kind): mask[:, layer, a:b].unsqueeze(0)
            for layer in range(mask.shape[1])
            for kind, a, b in [('attention', 0, query_heads), ('mlp', query_heads, mask.shape[-1])]}


@torch.no_grad()
def quality_rows(teacher_logits, student_logits, ids, chunk=32):
    """Next-token NLL and KL at positions 0..L-2, excluding last unused logit."""
    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 3 or teacher_logits.shape[:2] != ids.shape:
        raise ValueError('Logits and token IDs must align')
    if len(ids) != 1 or ids.shape[1] < 2:
        raise ValueError('Use one unpadded window of at least two tokens')
    if not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all():
        raise ValueError('Nonfinite logits')
    rows = {k: [] for k in ['teacher_nll', 'student_nll', 'kl', 'top1_agreement']}
    for begin in range(0, ids.shape[1]-1, chunk):
        end = min(begin+chunk, ids.shape[1]-1)
        a = teacher_logits[0, begin:end].float().log_softmax(-1)
        b = student_logits[0, begin:end].float().log_softmax(-1)
        target = ids[0, begin+1:end+1, None]
        rows['teacher_nll'].append(-a.gather(-1, target).squeeze(-1))
        rows['student_nll'].append(-b.gather(-1, target).squeeze(-1))
        kl = (a.exp() * (a-b)).sum(-1)
        if (kl < -1e-5).any():
            raise ValueError('KL is materially negative')
        rows['kl'].append(kl.clamp_min(0))
        rows['top1_agreement'].append((a.argmax(-1) == b.argmax(-1)).float())
    return {k: torch.cat(v).cpu() for k, v in rows.items()}


def concentration(energy):
    """Per row concentration; group energy is the SUM of constituent squares."""
    if energy.ndim < 2 or not torch.isfinite(energy).all() or (energy < 0).any():
        raise ValueError('Invalid nonnegative energy')
    ordered = energy.float().sort(-1, descending=True).values
    total = ordered.sum(-1, keepdim=True)
    cumulative = ordered.cumsum(-1)
    prior = cumulative - ordered
    positive = total.squeeze(-1) > 0
    result = {}
    for coverage in [.5, .9, .95]:
        counts = ((prior < coverage*total) & (total > 0)).sum(-1)
        result[f'fraction_for_{round(coverage*100)}pct_energy'] = counts.float()/energy.shape[-1]
    for fraction in [.1, .25, .5]:
        k = max(1, math.ceil(fraction*energy.shape[-1]))
        result[f'energy_in_top_{round(fraction*100)}pct'] = ordered[..., :k].sum(-1)/total.squeeze(-1).clamp_min(1e-30)
    norms = ordered.sqrt()
    norm_total = norms.sum(-1, keepdim=True)
    result['fraction_for_95pct_norm_sum'] = (((norms.cumsum(-1)-norms) < .95*norm_total) & (norm_total > 0)).sum(-1).float()/energy.shape[-1]
    result['nonzero_fraction'] = (energy > 0).float().mean(-1)
    result['nonzero_total'] = positive.float()
    return result
