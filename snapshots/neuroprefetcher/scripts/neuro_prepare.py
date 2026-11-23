"""Training-only percentile and centroid preparation, with resumable outputs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from pathpred.collection import atomic_json, atomic_tensors
from pathpred.neuroprefetcher import fit_centroids
from pathpred.neuro_features import capture_neuro
from scripts.neuron_guessing import backbone
from scripts.neuro_data import load_matching, window_ids


def collect(c):
    root = Path(c['root']); root.mkdir(parents=True, exist_ok=True)
    folder = root/'samples'; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists(): return
    sources, windows, audit = load_matching(c); atomic_json(root/'data-audit.json', audit)
    model, controller = backbone(); n = c['centroid_positions']; width = model.config.intermediate_size
    keys = [f'{projection}_{layer}' for layer in range(c['dense_prefix']-1, model.config.num_hidden_layers)
            for projection in ('gate','up')]
    arrays = {key: np.lib.format.open_memmap(folder/(key+'.npy'), mode='w+', dtype=np.float16,
                                            shape=(n,width)) for key in keys}
    seen = 0; started = time.monotonic()
    with torch.no_grad():
        for index, row in enumerate(windows):
            ids = window_ids(sources, row)
            with controller.use_gates(), capture_neuro(model) as trace: model.model(ids, use_cache=False)
            positions = torch.arange(0, ids.shape[1], c['centroid_sample_stride'], device=ids.device)[:n-seen]
            count = len(positions)
            for (layer, projection), value in trace['activations'].items():
                arrays[f'{projection}_{layer}'][seen:seen+count] = value[0,positions].float().cpu().numpy()
            seen += count
            if index%32 == 0:
                atomic_json(folder/'progress.json', {'positions': seen, 'target': n, 'seconds': time.monotonic()-started})
                print(json.dumps({'stage':'collection','positions':seen}), flush=True)
            if seen == n: break
    assert seen == n
    for array in arrays.values(): array.flush()
    atomic_json(folder/'complete.json', {'positions':seen,'units':width,'keys':keys,'seconds':time.monotonic()-started,
        'sample_source':'Initial matching training windows only; stride16; raw gate/up outputs'})


def cluster(c, budget):
    root = Path(c['root']); folder = root/f'budget-{round(100*budget)}'; folder.mkdir(exist_ok=True)
    sample = json.loads((root/'samples/complete.json').read_text())
    manifest = {}; started = time.monotonic()
    for key in sample['keys']:
        output = folder/(key+'.safetensors'); meta = output.with_suffix('.json')
        if output.exists() and meta.exists(): manifest[key] = json.loads(meta.read_text()); continue
        raw = np.load(root/'samples'/(key+'.npy'), mmap_mode='r')
        values = torch.tensor(np.asarray(raw), device='cuda', dtype=torch.float32).abs()
        rng = torch.Generator(device='cuda').manual_seed(c['seed'])
        indices = torch.randint(values.numel(), (262144,), generator=rng, device='cuda')
        threshold = values.flatten()[indices].quantile(1-budget)
        binary = values >= threshold; del values
        result = fit_centroids(binary, k=c['centroids'], keep=budget,
            iterations=c['centroid_iterations'], seed=c['seed'])
        file = atomic_tensors(output, {'aggregate':result['aggregate'].cpu(), 'masks':result['masks'].cpu(),
            'threshold':threshold.cpu(), 'counts':result['counts'].cpu()})
        record = {'file':file,'budget':budget,'unique_centroids':result['unique_centroids'],
                  'history':result['history'],'sample_positions':sample['positions']}
        atomic_json(meta, record); manifest[key] = record
        print(json.dumps({'stage':'clustering','key':key,'budget':budget,'unique':result['unique_centroids'],
                          'seconds':time.monotonic()-started}), flush=True)
        del binary, result; torch.cuda.empty_cache()
    atomic_json(folder/'centroids.json', {'banks':manifest,'budget':budget,'seconds':time.monotonic()-started})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',default='configs/neuroprefetcher.json')
    parser.add_argument('--collect',action='store_true');parser.add_argument('--budget',type=float)
    a=parser.parse_args();c=json.loads(Path(a.config).read_text())
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    if a.collect: collect(c)
    else:
        assert a.budget in c['nominal_budgets']; cluster(c,a.budget)


if __name__=='__main__': main()
