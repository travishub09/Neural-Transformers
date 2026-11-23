"""Causal, prefix-only features and separate raw gate/up target activations."""
from contextlib import contextmanager
import torch
from .neuroprefetcher import attention_sketch, nearest_centroid


@contextmanager
def capture_neuro(model, *, prefix=2, targets=True):
    """Hooks preserve native Qwen execution; no labels enter predictor features."""
    trace = {'features': {}, 'activations': {}}
    handles = []
    last = model.model.layers[prefix-1]
    def feature(name):
        def hook(module, args, output): trace['features'][name] = output.detach()
        return hook
    handles.append(model.model.embed_tokens.register_forward_hook(feature('embedding')))
    handles.append(last.input_layernorm.register_forward_hook(feature('pre_attention')))
    def attention(module, args, output):
        probabilities = output[1]
        if probabilities is None: raise RuntimeError('NeuroPrefetcher requires eager attention probabilities')
        trace['features']['sketch'] = attention_sketch(probabilities.detach())
    handles.append(last.self_attn.register_forward_hook(attention))
    def norm(module, args, output):
        # Matches the reference runtime: RMS of post-attention normalized state.
        trace['features']['post_attention_rms'] = output.detach().float().square().mean(-1, keepdim=True).sqrt()
    handles.append(last.post_attention_layernorm.register_forward_hook(norm))
    for layer_index, layer in enumerate(model.model.layers):
        if layer_index < prefix-1: continue
        for projection in ('gate', 'up'):
            if not targets and (layer_index != prefix-1 or projection != 'gate'): continue
            def record(module, args, output, key=(layer_index, projection)):
                trace['activations'][key] = output.detach()
            handles.append(getattr(layer.mlp, projection+'_proj').register_forward_hook(record))
    try: yield trace
    finally:
        for handle in handles: handle.remove()


def feature_tuple(trace):
    return tuple(trace['features'][key].float() for key in
                 ('embedding', 'pre_attention', 'sketch', 'post_attention_rms'))


def target_ids(trace, thresholds, banks, *, prefix=2, layers=28):
    """Dense targets: fixed training percentiles, per-layer/per-projection banks."""
    labels = []
    for projection in ('gate', 'up'):
        ids = []
        for layer in range(prefix, layers):
            key = f'{projection}_{layer}'
            mask = trace['activations'][(layer, projection)].abs() >= thresholds[key]
            ids.append(nearest_centroid(mask, banks[key]))
        labels.append(torch.stack(ids, -1))
    key = f'gate_{prefix-1}'
    context = nearest_centroid(trace['activations'][(prefix-1,'gate')].abs() >= thresholds[key], banks[key])
    return context, labels


def predicted_masks(gate_ids, up_ids, banks, *, prefix=2):
    result = {}
    for index in range(gate_ids.shape[-1]):
        layer = prefix + index
        result[layer] = banks[f'gate_{layer}'][gate_ids[...,index]] & banks[f'up_{layer}'][up_ids[...,index]]
    return result
