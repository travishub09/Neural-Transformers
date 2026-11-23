"""Cached-activation Phase 0 training. No LLM weights or test selection involved."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from pathpred.collection import atomic_json, file_sha256
from pathpred.path_model import PathConfig, PathModel
from pathpred.phase0 import budget_plan, mask_for_budget, select_units, weighted_loss, metric_rows, summarize_rows


class PhaseData:
    def __init__(self, root, device='cuda', include_test=False, target='coverage'):
        root = Path(root)
        m = json.loads((root / 'manifest.json').read_text())
        assert m['status'] == 'complete'
        for entry in m['files'].values():
            assert file_sha256(root / entry['file']) == entry['sha256']
        self.manifest = m
        self.device = device
        self.t = load_file(str(root / m['files']['examples']['file']), device=device)
        if target == 'strong':
            self.t['labels'] = self.t['binary'] == 1
        elif target != 'coverage':
            raise ValueError('Unknown prediction target')
        h = load_file(str(root / m['files']['hidden']['file']), device=device)
        self.hidden = h['hidden'].float() / h['rms']
        self.n, self.layers, self.units = self.t['binary'].shape
        self.ids = {}
        for split, code in [('train', 0), ('validation', 1), ('test', 2)]:
            if split == 'test' and not include_test:
                continue
            self.ids[split] = ((self.t['split_codes'] == code) &
                               (self.t['previous_sample_indices'] >= 0)).nonzero().flatten()

    def mask(self, counts, seed):
        return mask_for_budget(self.n, self.layers, self.units, counts, seed, self.device)

    def inputs(self, ids, mask, signal):
        prev = self.t['previous_sample_indices'][ids]
        if (prev < 0).any():
            raise ValueError('Phase 0 requires adjacent paired positions')
        valid = torch.ones(len(ids), self.layers, dtype=torch.bool, device=self.device)
        if signal == 'hidden':
            return {'hidden': self.hidden[ids], 'previous_hidden': self.hidden[prev], 'previous_valid': valid}
        key = 'binary' if signal == 'decisions' else signal
        return {'states': torch.where(mask[ids], self.t[key][ids], -1),
                'previous': torch.where(mask[prev], self.t[key][prev], -1), 'previous_valid': valid}


def model_config(signal):
    return PathConfig(**{**json.loads(Path('configs/path-model.json').read_text())['default'], 'signal': signal})


def accounting_and_plans(cfg, costs):
    accounting = {s: PathModel(model_config(s)).accounting() for s in cfg['signals']}
    ref = accounting['binary']['matmul_flops_per_llm_token']
    plans = {str(b): {s: budget_plan(costs, b, a['matmul_flops_per_llm_token'], ref)
                      for s, a in accounting.items()} for b in cfg['gated_budget_anchors']}
    for b in cfg['gated_budget_anchors']:
        plans[str(b)]['static'] = budget_plan(costs, b, 0, ref)
    return accounting, plans


def run_name(signal, budget, seed):
    return f'{signal}-b{round(budget*100):02d}-s{seed}'


@torch.no_grad()
def evaluate_model(model, data, split, mask, counts, cfg):
    model.eval()
    batches = []
    loss_sum, n = 0., 0
    for ids in data.ids[split].split(cfg['batch_size']):
        logits = model(**data.inputs(ids, mask, model.config.signal))
        labels = data.t['labels'][ids, 2:]
        loss_sum += float(weighted_loss(logits, labels, cfg['positive_loss_weight'])) * len(ids)
        n += len(ids)
        rows = metric_rows(select_units(logits, counts), labels,
            data.t['shares'][ids, 2:], data.t['binary'][ids, 2:] == 1)
        batches.append({k: v.cpu().numpy() for k, v in rows.items()})
    rows = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    metrics = summarize_rows(rows)
    metrics['loss'] = loss_sum / n
    return metrics, rows


def training_fingerprint(dataset, cfg, config_path=Path('configs/phase0.json')):
    files = ['pathpred/phase0.py', 'pathpred/phase0_training.py', 'pathpred/path_model.py',
             'scripts/train_phase0.py', str(config_path), 'configs/path-model.json',
             cfg.get('protocol_path', 'docs/phase0-protocol.md')]
    return {'files': {f: file_sha256(Path(f)) for f in files}, 'config': cfg,
            'dataset_manifest_sha256': file_sha256(Path(dataset) / 'manifest.json')}


def save_torch_atomic(path, content):
    temporary = path.with_suffix('.tmp')
    torch.save(content, temporary)
    temporary.replace(path)


def train_one(data, signal, budget, seed, cfg, plan, mask_counts, output, fingerprint, deadline):
    folder = Path(output) / run_name(signal, budget, seed)
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / 'report.json'
    if report_path.exists():
        old = json.loads(report_path.read_text())
        assert old['fingerprint'] == fingerprint
        assert file_sha256(folder / 'best.safetensors') == old['checkpoint_sha256']
        return old
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PathModel(model_config(signal)).to(data.device)
    train_ids = data.ids['train']
    prior = (data.t['labels'][train_ids, 2:].float().sum(0) + 1) / (len(train_ids) + 2)
    with torch.no_grad():
        model.output_bias.copy_((cfg['positive_loss_weight'] * prior / (1-prior)).log())
        model.output_weight.mul_(cfg['output_initialization_scale'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history, best, stale, start_epoch = [], None, 0, 0
    last_path = folder / 'last.pt'
    if last_path.exists():
        last = torch.load(last_path, map_location=data.device, weights_only=True)
        assert last['fingerprint'] == fingerprint
        model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer'])
        history, best, stale, start_epoch = last['history'], last['best'], last['stale'], last['next_epoch']
    validation_mask = data.mask(mask_counts, cfg['validation_mask_seed'])
    started = time.monotonic()
    steps_per_epoch = math.ceil(len(train_ids) / cfg['batch_size'])
    for epoch in range(start_epoch, cfg['epochs']):
        if time.monotonic() >= deadline:
            raise TimeoutError('Cumulative GPU training guard reached; checkpoints retained')
        model.train()
        # Seeds deliberately independent of signal/budget and model initialization.
        mask = data.mask(mask_counts, seed * 10000 + epoch)
        gen = torch.Generator(device=data.device).manual_seed(seed * 10000 + epoch + 5000000)
        order = train_ids[torch.randperm(len(train_ids), generator=gen, device=data.device)]
        train_loss = 0.
        epoch_start = time.monotonic()
        for bi, ids in enumerate(order.split(cfg['batch_size'])):
            step = epoch * steps_per_epoch + bi
            warmup = cfg['warmup_epochs'] * steps_per_epoch
            total = cfg['epochs'] * steps_per_epoch
            if step < warmup:
                factor = (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, total-warmup-1)
                factor = cfg['minimum_lr_fraction'] + (1-cfg['minimum_lr_fraction']) * (1+math.cos(math.pi*progress))/2
            for group in optimizer.param_groups:
                group['lr'] = cfg['learning_rate'] * factor
            optimizer.zero_grad(set_to_none=True)
            logits = model(**data.inputs(ids, mask, signal))
            loss = weighted_loss(logits, data.t['labels'][ids, 2:], cfg['positive_loss_weight'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            train_loss += float(loss.detach()) * len(ids)
        val, _ = evaluate_model(model, data, 'validation', validation_mask, plan['counts'], cfg)
        score = (val['recall'], -val['loss'])
        improved = best is None or score > (best['recall'], -best['loss'])
        if improved:
            best = {'epoch': epoch+1, 'recall': val['recall'], 'loss': val['loss'], 'metrics': val}
            temp = folder / 'best.tmp'
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temp))
            temp.replace(folder / 'best.safetensors')
            stale = 0
        else:
            stale += 1
        record = {'epoch': epoch+1, 'training_loss': train_loss / len(train_ids),
                  'validation': val, 'seconds': time.monotonic()-epoch_start, 'improved': improved}
        history.append(record)
        save_torch_atomic(last_path, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'fingerprint': fingerprint, 'history': history, 'best': best, 'stale': stale, 'next_epoch': epoch+1})
        atomic_json(folder / 'progress.json', {'run': folder.name, 'history': history, 'best': best})
        print(json.dumps({'run': folder.name, 'epoch': epoch+1, 'validation_recall': val['recall'],
                          'loss': val['loss'], 'epoch_seconds': record['seconds']}), flush=True)
        if epoch+1 >= cfg['minimum_epochs'] and stale >= cfg['early_stopping_patience']:
            break
    report = {'run': folder.name, 'signal': signal, 'budget': budget, 'seed': seed,
              'fingerprint': fingerprint, 'model_config': asdict(model.config), 'plan': plan,
              'accounting': model.accounting(), 'best': best, 'history': history,
              'training_seconds': sum(r['seconds'] for r in history),
              'invocation_seconds': time.monotonic()-started,
              'checkpoint_sha256': file_sha256(folder / 'best.safetensors')}
    atomic_json(report_path, report)
    del model, optimizer
    torch.cuda.empty_cache()
    return report
