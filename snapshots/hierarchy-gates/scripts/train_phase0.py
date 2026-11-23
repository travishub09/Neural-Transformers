import argparse
import json
from pathlib import Path
import time
import torch
from pathpred.collection import atomic_json
from pathpred.phase0_training import PhaseData, accounting_and_plans, training_fingerprint, train_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/stage6-phase0-v1'))
    p.add_argument('--output', type=Path, default=Path('artifacts/phase0-v1'))
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--config', type=Path, default=Path('configs/phase0.json'))
    a = p.parse_args()
    if (a.output / 'freeze.json').exists():
        raise ValueError('Experiment frozen; cannot train into this output')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads(a.config.read_text())
    if cfg.get('target') == 'strong' and a.output.name == 'phase0-v1':
        raise ValueError('Strong-target experiment needs a separate output directory')
    if a.pilot:
        if 'pilot' not in a.output.name:
            raise ValueError('Pilot requires a separate pilot-named output directory')
        cfg.update(epochs=2, minimum_epochs=2, signals=['decisions', 'binary', 'coarse', 'magnitudes', 'hidden'])
    costs = json.loads(Path('data/stage4-clusters-v1/unit-costs.json').read_text())
    accounting, plans = accounting_and_plans(cfg, costs)
    fp = training_fingerprint(a.data, cfg, a.config)
    a.output.mkdir(parents=True, exist_ok=True)
    manifest_path = a.output / 'training.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old:
        assert old['fingerprint'] == fp
    elapsed_before = old.get('cumulative_seconds', 0) if old else 0
    manifest = {'status': 'in_progress', 'fingerprint': fp, 'pilot': a.pilot,
                'accounting': accounting, 'plans': plans, 'cumulative_seconds': elapsed_before,
                'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__)}
    atomic_json(manifest_path, manifest)
    data = PhaseData(a.data, target=cfg.get('target', 'coverage'))
    start = time.monotonic()
    deadline = start + cfg['training_gpu_time_limit_hours'] * 3600 - elapsed_before
    jobs = [('binary', .25, 101)] if a.pilot else [
        (s, b, seed) for b in cfg['gated_budget_anchors'] for seed in cfg['seeds'] for s in cfg['signals']]
    try:
        for signal, budget, seed in jobs:
            train_one(data, signal, budget, seed, cfg, plans[str(budget)][signal],
                      plans[str(budget)]['binary']['counts'], a.output, fp, deadline)
            manifest['cumulative_seconds'] = elapsed_before + time.monotonic()-start
            atomic_json(manifest_path, manifest)
        manifest['status'] = 'complete'
    finally:
        manifest['cumulative_seconds'] = elapsed_before + time.monotonic()-start
        atomic_json(manifest_path, manifest)
    print(json.dumps({'status': manifest['status'], 'seconds': manifest['cumulative_seconds'],
                      'runs': len(jobs), 'test_evaluated': False}), flush=True)


if __name__ == '__main__':
    main()
