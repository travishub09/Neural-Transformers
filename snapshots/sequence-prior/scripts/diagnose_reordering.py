"""Separate BF16 reduction-policy effects from permutation correctness."""
import argparse
import copy
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from pathpred.clustering import reorder_mlp_
from pathpred.collection import atomic_json
from scripts.verify_reordered_model import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.load(open('configs/qwen3-1.7b-base.json'))
    path = snapshot_download(cfg['model_id'], revision=cfg['revision'], local_files_only=True)
    original = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
        device_map={'': 'cuda:0'}, attn_implementation='eager', local_files_only=True).eval().requires_grad_(False)
    model = copy.deepcopy(original)
    perms = load_file('data/stage4-clusters-v1/permutations.safetensors')['permutations']
    for layer, perm in zip(model.model.layers, perms):
        reorder_mlp_(layer.mlp, perm)
    rows = [json.loads(x) for x in Path('data/stage3-corpus-v1/sequences.jsonl').read_text().splitlines()]
    r = next(x for x in rows if x['domain'] == 'web' and x['split'] == 'test')
    ids = load_file('data/stage3-corpus-v1/tokens.safetensors')['input_ids'][r['sequence_id']:r['sequence_id'] + 1, :r['length']].long().cuda()
    report = {'sequence_id': r['sequence_id'], 'results': {}}
    default_logits = None
    for mode in ('bf16_reduced', 'bf16_full_accumulation', 'float32'):
        dtype = torch.float32 if mode == 'float32' else torch.bfloat16
        original.to(dtype); model.to(dtype)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = mode == 'bf16_reduced'
        with torch.inference_mode():
            a = original(ids, use_cache=False, output_hidden_states=True)
            b = model(ids, use_cache=False, output_hidden_states=True)
            metrics = compare(a.logits, b.logits)
            metrics['hidden_max_abs_by_layer'] = [float((x.float() - y.float()).abs().max()) for x, y in zip(a.hidden_states, b.hidden_states)]
            if default_logits is None:
                default_logits = a.logits.cpu()
            else:
                metrics['default_dense_to_this_dense'] = compare(default_logits.to('cuda'), a.logits)
                metrics['default_dense_to_this_reordered'] = compare(default_logits.to('cuda'), b.logits)
        report['results'][mode] = metrics
        atomic_json(args.output, report)
        print(json.dumps({'mode': mode, **metrics}), flush=True)
        del a, b


if __name__ == '__main__':
    main()
