"""Greedy draft/verify with separate target KV and rollback of planner history.

This is a greedy specialization, not the rejection-sampling algorithm for a
nonzero-temperature target. The target consumes one pending token plus a draft
block per verification. Draft and target share weights, never KV state.
"""
import time
import torch


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def crop_cache(cache, length):
    cache.crop(length)
    if cache.get_seq_length() != length:
        raise RuntimeError('Cache rollback failed')


@torch.inference_mode()
def dense_greedy(policy, prompt, count, eos=None):
    device = next(policy.model.parameters()).device
    ids = torch.tensor([prompt], device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    policy.reset(); sync(device); start = time.perf_counter()
    logits = policy.prefill(ids, valid)
    sync(device); prefill = time.perf_counter() - start
    start = time.perf_counter(); result = []
    for i in range(count):
        token = int(logits.argmax(-1).item()); result.append(token)
        if token == eos or i + 1 == count:
            break
        valid = torch.ones(1, len(prompt) + len(result), dtype=torch.bool, device=device)
        logits = policy.step(torch.tensor([[token]], device=device), valid)
    sync(device)
    return dict(tokens=result, prefill_seconds=prefill,
                decode_seconds=time.perf_counter()-start)


@torch.inference_mode()
def speculative_greedy(draft, prompt, count, block, eos=None):
    if len(prompt) < 2 or count < 1 or block < 1:
        raise ValueError('Need at least two prompt tokens and positive lengths')
    model, ctrl = draft.model, draft.ctrl
    device = next(model.parameters()).device
    def tensor(x): return torch.tensor([x], device=device)
    def mask(n): return torch.ones(1, n, dtype=torch.bool, device=device)
    def forward(seq, cache, old):
        with ctrl.use_gates({}):
            return model(tensor(seq), attention_mask=mask(old+len(seq)),
                         position_ids=torch.arange(old, old+len(seq), device=device)[None],
                         past_key_values=cache, use_cache=True)
    draft.reset(); sync(device); start = time.perf_counter()
    logits = draft.prefill(tensor(prompt), mask(len(prompt)))
    target = forward(prompt[:-1], None, 0).past_key_values
    sync(device); prefill = time.perf_counter()-start
    pending = prompt[-1]; result = []; steps = []
    retention_sum=torch.zeros((), device=device); sparse_steps=0
    def retain():
        nonlocal retention_sum, sparse_steps
        gates=draft.masks()
        if gates is not None:
            retention_sum += gates.float().mean(); sparse_steps += 1
    start = time.perf_counter()
    while len(result) < count:
        committed = len(prompt) + len(result)
        proposed = []; states = [draft.previous]
        # The extra dense correction/bonus can be clipped by the output cap.
        k = min(block, count-len(result))
        for j in range(k):
            token = int(logits.argmax(-1).item()); proposed.append(token)
            logits = draft.step(tensor([token]), mask(committed+j+1))
            retain()
            states.append(draft.previous)
        old = target.get_seq_length()
        out = forward([pending]+proposed, target, old)
        target = out.past_key_values
        choices = out.logits[0].argmax(-1).tolist()
        accepted = 0
        while accepted < k and choices[accepted] == proposed[accepted]:
            accepted += 1
        emitted = proposed[:accepted] + [choices[accepted]]
        emitted = emitted[:count-len(result)]
        terminal = eos is not None and eos in emitted
        if terminal: emitted = emitted[:emitted.index(eos)+1]
        result.extend(emitted)
        steps.append(dict(proposed=k, accepted=accepted, emitted=len(emitted)))
        if terminal or len(result) == count: break
        # Keep the accepted prefix. Correction is the next target's pending token.
        crop_cache(target, old+1+accepted)
        crop_cache(draft.cache, committed+accepted)
        draft.previous = states[accepted]
        pending = choices[accepted]
        logits = draft.step(tensor([pending]), mask(committed+accepted+1))
        retain()
    sync(device)
    seconds=time.perf_counter()-start
    return dict(tokens=result, steps=steps, prefill_seconds=prefill,
                decode_seconds=seconds, draft_sparse_executions=sparse_steps,
                draft_retention=float(retention_sum.item())/max(1,sparse_steps))
