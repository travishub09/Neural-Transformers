"""Training primitives for a whole-suffix planner; Qwen remains frozen."""
from contextlib import contextmanager
import torch
from .neuron_prediction import collect_dense, recall_weighted_loss
from .planner_route import planner_route, previous_context, record_parallel_paths
from .end_to_end import dense_teacher, quality_loss


@torch.no_grad()
def copy_targets(model, controller, thresholds, ids, keep, prefix=2):
    """Teacher labels and the *raw* dense-prefix state, produced on the fly.

    The previous-token warm-start history is privileged teacher forcing: dense
    firing bits masked by dense magnitude selections, not an on-policy rollout.
    No target mask for the current layer is ever an input to its prediction.
    """
    prefix_state = {}
    def hook(module, args, output):
        prefix_state['hidden'] = (output[0] if isinstance(output, tuple) else output).detach()
    handle = model.model.layers[prefix-1].register_forward_hook(hook)
    try:
        with controller.use_gates(), collect_dense(model, thresholds, budgets=(keep,), capture_hidden=False) as trace:
            model.model(ids, use_cache=False)
    finally:
        handle.remove()
    labels = torch.stack([trace['labels'][l][str(keep)] for l in range(prefix, len(model.model.layers))], 2)
    observed = torch.stack([trace['bits'][l] if l < prefix else
        torch.where(trace['labels'][l][str(keep)], trace['bits'][l], -1)
        for l in range(len(model.model.layers))], 2)
    previous, valid = previous_context(observed)
    return prefix_state['hidden'], labels, previous, valid


def copy_batch(model, controller, planner, optimizer, thresholds, ids, *, keep,
               teacher_probability=1., scheduled=False, positive_weight=2.):
    planner.train(); optimizer.zero_grad(set_to_none=True)
    hidden, labels, previous, valid = copy_targets(model, controller, thresholds, ids, keep,
                                                   planner.config.dense_prefix)
    if not planner.config.previous_path:
        previous = valid = None
    elif scheduled:
        # This no-history bootstrap is a two-pass approximation, deliberately
        # not described as the true sequential deployment history.
        planner.eval()
        previous, valid = record_parallel_paths(model, controller, planner, thresholds, ids, keep=keep)
        planner.train()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=ids.is_cuda):
        if scheduled:
            result = planner.plan(hidden, previous, valid, mode='topk', keep=keep,
                teacher_masks=labels, teacher_probability=teacher_probability)
            scores = result.scores
        else:
            scores = planner.teacher_forced(hidden, labels, previous, valid)
        loss = recall_weighted_loss(scores, labels, positive_weight)
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite copy loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(planner.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return {'loss': float(loss.detach()), 'gradient_norm': float(norm),
            'teacher_feedback_probability': teacher_probability if scheduled else 1.,
            'history': ('none' if not planner.config.previous_path else
                        'bootstrap two-pass' if scheduled else 'teacher-forced dense magnitude path')}


def end_to_end_batch(model, controller, planner, optimizer, thresholds, ids, offsets, dual,
                     *, mode='gumbel', temperature=1., beta=1., epsilon=.01, chunk_size=32):
    optimizer.zero_grad(set_to_none=True); planner.eval()
    previous = valid = None
    if planner.config.previous_path:
        previous, valid = record_parallel_paths(model, controller, planner, thresholds, ids,
                                                offsets=offsets, keep=dual.target)
    teacher, teacher_pre = dense_teacher(model, controller, ids)
    planner.train()
    with planner_route(model, controller, planner, thresholds, previous=previous,
            previous_valid=valid, mode=mode, temperature=temperature,
            offsets=offsets, keep=dual.target) as trace:
        hidden = model.model(ids, use_cache=False).last_hidden_state
        loss, metrics = quality_loss(model, hidden, trace['pre_norm'], teacher, teacher_pre, trace,
            alpha=dual.alpha, beta=beta, epsilon=epsilon, chunk_size=chunk_size)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite planner quality loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(planner.parameters(), 1., error_if_nonfinite=True)
        assert trace['predictor_calls'] == 1
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        optimizer.step()
        result = {k: float(v.detach()) for k, v in metrics.items()}
        result.update(loss=float(loss.detach()), gradient_norm=float(norm), alpha=dual.alpha,
                      whole_plan_calls=trace['predictor_calls'], recomputations=trace['recomputations'],
                      history='no-history bootstrap two-pass approximation' if planner.config.previous_path else 'none')
    dual.update(result['used'])
    return result
