"""Output-quality training through a frozen, causally gated Qwen backbone.

Activity bits and observed flags are deliberately stop-gradient. Gradients flow
through Qwen's arithmetic into its gates, and through the predictor's own cache.
No teacher features enter the predictor. This is dense quality simulation.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .sparsity_diagnostics import select_mask


@dataclass
class DualBudget:
    target: float = .5
    alpha: float = 1.
    learning_rate: float = .05
    maximum: float = 100.

    def update(self, used):
        self.alpha = min(self.maximum, max(0., self.alpha + self.learning_rate * (float(used) - self.target)))
        return self.alpha


def neuron_gates(scores, temperature=1., mode='soft', keep=.5):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    soft = torch.sigmoid(scores.float() / temperature)
    if mode == 'soft':
        gates = soft
    elif mode == 'hard':
        hard = (scores >= 0).to(soft.dtype)
        gates = hard + (soft - soft.detach())
    elif mode == 'topk':
        gates = select_mask(scores, keep).to(soft.dtype)
    elif mode == 'dense':
        gates = torch.ones_like(soft)
    else:
        raise ValueError(mode)
    # Soft nonzero values below .5 carry no observation to later predictors.
    observed = gates.detach() >= .5
    return gates, soft, observed


@contextmanager
def quality_route(model, controller, predictor, thresholds, *, temperature=1.,
                  mode='soft', keep=.5, checkpoint_layers=True, score_offsets=None):
    """Keep this context open through backward; each layer's gates are immutable.

Only the decoder's original forward is recomputed. Its outer post-forward hook
predicts once, so backward never generates a new routing decision. Gate tensors
are explicit checkpoint inputs. Hooks suppress observation writes on recompute.
"""
    if controller.state.active:
        raise RuntimeError('quality_route owns its gate context')
    trace = {'bits': {}, 'gates': {}, 'soft': {}, 'scores': {}, 'pre_norm': None,
             'predictor_calls': 0, 'recomputations': 0}
    handles = []; originals = []; cache = None; recomputing = False

    @contextmanager
    def recompute_context():
        nonlocal recomputing
        previous = recomputing; recomputing = True
        trace['recomputations'] += 1
        try:
            yield
        finally:
            recomputing = previous

    def checkpoint_forward(layer_id, original):
        def wrapped(hidden_states, *args, **kwargs):
            gate = controller.state.gates.get((layer_id, 'mlp'))
            if gate is None or not torch.is_grad_enabled():
                return original(hidden_states, *args, **kwargs)
            def run(hidden, fixed_gate):
                if not controller.state.active:
                    raise RuntimeError('Backward must stay inside quality_route')
                controller.state.gates[layer_id, 'mlp'] = fixed_gate
                return original(hidden, *args, **kwargs)
            return checkpoint(run, hidden_states, gate, use_reentrant=False,
                              context_fn=lambda: (nullcontext(), recompute_context()))
        return wrapped

    def observe(layer_id):
        def hook(module, args):
            if recomputing:
                return
            activation = args[0].detach().float().abs()
            if layer_id < predictor.config.lookahead:
                observed = torch.ones_like(activation, dtype=torch.bool)
                trace['gates'][layer_id] = observed
            else:
                observed = trace['gates'][layer_id].detach() >= .5
            trace['bits'][layer_id] = torch.where(
                observed, (activation > thresholds[layer_id]).to(torch.int8), -1)
        return hook

    def predict(layer_id):
        def hook(module, args, output):
            nonlocal cache
            if layer_id >= predictor.steps:
                return
            state = trace['bits'][layer_id]
            inputs = {}
            if predictor.config.signal in ('hidden', 'combined'):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.detach().float()
                hidden = (hidden / hidden.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)).to(torch.bfloat16)
                inputs['hidden'] = hidden
            if predictor.config.signal != 'hidden':
                inputs['states'] = state
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=state.is_cuda):
                prediction = predictor.step_tokens(**inputs, cache=cache)
            cache = prediction.cache
            target = prediction.target_layer
            scores = prediction.logits
            if score_offsets is not None and mode not in ('topk', 'dense'):
                scores = scores - score_offsets[target]
            gates, soft, _ = neuron_gates(scores, temperature, mode, keep)
            trace['scores'][target] = prediction.logits.detach()
            trace['gates'][target] = gates
            trace['soft'][target] = soft
            controller.state.gates[target, 'mlp'] = gates
            trace['predictor_calls'] += 1
        return hook

    def final_hidden(module, args):
        trace['pre_norm'] = args[0]

    try:
        with controller.use_gates():
            for i, layer in enumerate(model.model.layers):
                if checkpoint_layers:
                    originals.append((layer, layer.forward))
                    layer.forward = checkpoint_forward(i, layer.forward)
                handles.append(layer.mlp.down_proj.register_forward_pre_hook(observe(i)))
                handles.append(layer.register_forward_hook(predict(i)))
            handles.append(model.model.norm.register_forward_pre_hook(final_hidden))
            yield trace
    finally:
        for handle in handles:
            handle.remove()
        for layer, original in originals:
            layer.forward = original


@torch.no_grad()
def dense_teacher(model, controller, ids):
    record = {}
    def hook(module, args):
        record['pre_norm'] = args[0].detach()
    handle = model.model.norm.register_forward_pre_hook(hook)
    try:
        with controller.use_gates():
            logits = model(ids, use_cache=False).logits.detach()
        return logits, record['pre_norm']
    finally:
        handle.remove()


def quality_loss(model, student_hidden, student_pre_norm, teacher_logits,
                 teacher_pre_norm, trace, *, alpha=1., beta=1., epsilon=.01,
                 chunk_size=32, checkpoint_head=True,token_mask=None):
    """Exact full-vocabulary KL, chunked to bound the backward working set."""
    student = student_hidden.reshape(-1, student_hidden.shape[-1])
    teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    if token_mask is not None:
        if token_mask.shape!=student_hidden.shape[:2] or token_mask.dtype!=torch.bool:raise ValueError('Invalid loss mask')
        student=student[token_mask.reshape(-1)];teacher=teacher[token_mask.reshape(-1)]
    count = student.shape[0]
    if not count:raise ValueError('No scored training positions')
    def chunk_kl(hidden, targets):
        # Keep the head's numerical path identical to the frozen dense teacher.
        logits = model.lm_head(hidden).float()
        teacher_log = F.log_softmax(targets.float(), dim=-1)
        student_log = F.log_softmax(logits, dim=-1)
        return (teacher_log.exp() * (teacher_log - student_log)).sum() / count
    pieces = []
    for i in range(0, count, chunk_size):
        args = (student[i:i+chunk_size], teacher[i:i+chunk_size])
        pieces.append(checkpoint(chunk_kl, *args, use_reentrant=False)
                      if checkpoint_head and torch.is_grad_enabled() else chunk_kl(*args))
    kl = torch.stack(pieces).sum()
    cosine = F.cosine_similarity(student_pre_norm.float(), teacher_pre_norm.float(), dim=-1)
    if token_mask is not None:cosine=cosine[token_mask]
    log_cosine = -torch.log(cosine.clamp(min=epsilon, max=1.)).mean()
    # Equal-size individual neurons have equal MLP costs. Prefix is reported
    # separately; attention and predictor overhead are not included in this term.
    def selected(g):return g if token_mask is None else g[token_mask]
    routed = [selected(g).float().mean() for l, g in trace['gates'].items() if l >= 2]
    used = torch.stack(routed).mean()
    soft_mass = torch.stack([selected(g).mean() for g in trace['soft'].values()]).mean()
    hard_used = torch.stack([(selected(g).detach() >= .5).float().mean()
                             for l, g in trace['gates'].items() if l >= 2]).mean()
    loss = log_cosine + beta * kl + alpha * used
    return loss, {'log_cosine': log_cosine, 'kl': kl, 'used': used,
                  'soft_mass': soft_mass, 'hard_used': hard_used,
                  'cosine': cosine.mean(), 'quality': log_cosine + beta * kl}
