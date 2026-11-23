"""Retention-only HellaSwag train calibration, then untouched validation scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from pathpred.collection import atomic_json, file_sha256
from pathpred.budget_calibration import calibrate_threshold
from scripts.mistral_comparisons import ROOT, load, apply_calibration
from scripts.mistral_comparison_core import score_batch
from scripts.prepare_mistral_comparisons import process, encode_pair

OUT = ROOT / 'hellaswag-matched-v1'
CAL = Path('data/hellaswag-matched-calibration-v1')
TARGETS = {50: .48, 40: .40}
SEED = 20261115
MATCH_TOLERANCE = .005  # 0.5 percentage points on the full validation split


def select_indices(n, count=1024):
    return sorted(range(n), key=lambda i: hashlib.sha256(f'{SEED}:{i}'.encode()).hexdigest())[:count]


def checkpoint_identity(budget):
    assert budget in (50,40,30,20)
    family='mistral-lower-v1' if budget in (30,20) else 'mistral-phased-v1'
    config='mistral-lower-budgets.json' if budget in (30,20) else 'mistral-planner-phased.json'
    root = Path(f'artifacts/{family}/budget-{budget}')
    done = json.loads((root / 'completion.json').read_text())
    metadata = root / f"checkpoint-{done['tokens']:08d}.json"
    meta = json.loads(metadata.read_text())
    weight = root / meta['file']['file']
    assert file_sha256(weight) == meta['file']['sha256']
    paths = [root / 'completion.json', metadata, weight,
        root / f"check-{done['tokens']:08d}" / 'validation.json',
        Path('artifacts/mistral-pilot-v1/thresholds.safetensors'),
        Path('configs')/config]
    return {str(p): file_sha256(p) for p in paths}


def prepare():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    CAL.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path('data/hellaswag-mistral-v1/manifest.json').read_text())
    manifest = CAL / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old['revision'] == source['revision'] and old['seed'] == SEED and old['examples'] == 1024
        assert old['file_sha256'] == file_sha256(CAL / 'examples.jsonl')
        atomic_json(OUT / 'calibration-data.json', old)
        return
    ds = load_dataset('Rowan/hellaswag', revision=source['revision'], split='train')
    tok = AutoTokenizer.from_pretrained('checkpoints/mistral-7b-v0.1', local_files_only=True)
    heldout = {x['query'] for x in map(json.loads, Path('data/hellaswag-mistral-v1/examples.jsonl').read_text().splitlines())}
    chosen = []
    for i in select_indices(len(ds), len(ds)):
        # Do not read or persist answer labels in calibration artifacts.
        doc = dict(ds[i]);doc['label'] = 0
        row = process(doc);row.pop('gold')
        if row['query'] in heldout: continue
        row.update(id=i, sequences=[], prompt_lengths=[])
        for ending in row['choices']:
            seq, n = encode_pair(tok, row['query'], ' ' + ending)
            row['sequences'].append(seq);row['prompt_lengths'].append(n)
        chosen.append(row)
        if len(chosen) == 1024: break
    assert len(chosen) == 1024
    tmp = CAL / 'examples.tmp'
    tmp.write_text(''.join(json.dumps(x)+'\n' for x in chosen));tmp.replace(CAL / 'examples.jsonl')
    atomic_json(manifest, dict(dataset='Rowan/hellaswag', revision=source['revision'], split='train',
        examples=len(chosen), seed=SEED, selected_ids=[x['id'] for x in chosen], labels_used=False,
        selection='SHA256(seed:index) order; exclude queries present in validation',
        objective='executed answer-token eligible-MLP retention only; no accuracy or likelihood selection',
        file_sha256=file_sha256(CAL / 'examples.jsonl')))
    atomic_json(OUT / 'calibration-data.json', json.loads(manifest.read_text()))


def calibrate(budget):
    path = OUT / f'calibration-{budget}.json'
    identity = dict(checkpoint=checkpoint_identity(budget), corpus_sha256=file_sha256(CAL / 'examples.jsonl'),
        code={s:file_sha256(Path(s)) for s in ['scripts/mistral_hellaswag_matched.py','scripts/mistral_comparison_core.py',
            'scripts/mistral_comparisons.py','pathpred/budget_calibration.py']}, target=TARGETS[budget])
    identity_path = OUT / f'calibration-{budget}-identity.json'
    if identity_path.exists(): assert json.loads(identity_path.read_text()) == identity
    else: atomic_json(identity_path, identity)
    if path.exists(): return
    pol = load('planner', budget)
    if budget in (50,40):apply_calibration(pol, budget, 64)
    base = pol.offsets.clone()
    rows = [json.loads(x) for x in (CAL / 'examples.jsonl').read_text().splitlines()]
    assert all('gold' not in x for x in rows)
    rows.sort(key=lambda x: (max(map(len,x['sequences'])),x['id']))
    trials_path = OUT / f'calibration-{budget}-trials.json'
    trials = json.loads(trials_path.read_text()) if trials_path.exists() else {}
    def measure(shift):
        key = str(shift)
        if key in trials: return trials[key]
        pol.offsets = base + shift
        kept = executions = 0.
        for start in range(0, len(rows), 4):
            batch = rows[start:start+4]
            z = score_batch(pol, [s for x in batch for s in x['sequences']],
                [n for x in batch for n in x['prompt_lengths']], teacher=False)
            kept += float(z['kept_sum'].sum());executions += float(z['executions'].sum())
            print(json.dumps(dict(stage='retention_calibration',budget=budget,shift=shift,
                complete=start+len(batch),total=len(rows))),flush=True)
        assert executions > 0
        trials[key] = kept/executions;atomic_json(trials_path,trials)
        return trials[key]
    result = calibrate_threshold(measure, TARGETS[budget], tolerance=.001, max_evaluations=16)
    atomic_json(path, dict(checkpoint_budget=budget, target=TARGETS[budget], checkpoint=identity['checkpoint'],
        offsets=(base+result['shift']).cpu().tolist(),calibration=result,split='train',examples=len(rows),
        labels_used=False,identity=identity,full_validation_match_tolerance=MATCH_TOLERANCE))


def paired(first, second):
    a = {x['id']:x for x in first};b = {x['id']:x for x in second}
    assert a.keys() == b.keys() and len(a) == 10042
    assert all(a[i]['gold'] == b[i]['gold'] for i in a)
    delta = np.array([a[i]['accuracy_normalized']-b[i]['accuracy_normalized'] for i in sorted(a)])
    values, counts = np.unique(delta, return_counts=True)
    draws = np.random.default_rng(SEED).multinomial(len(delta),counts/len(delta),size=10000)@values/len(delta)
    return dict(gap_percentage_points=100*float(delta.mean()),ci95_percentage_points=(100*np.quantile(draws,[.025,.975])).tolist())


def references_for(budget):
    if budget in (50,40):
        return {name:ROOT/'hellaswag'/name for name in ['dense',f'griffin-{budget}',f'neuro-{budget}']}
    refs={f'{kind}-{b}':ROOT/'hellaswag'/f'{kind}-{b}' for kind in ['griffin','neuro'] for b in [50,40]}
    refs.update({'dense-A100':ROOT/'hellaswag/dense',
        'dense-H200':ROOT/'hellaswag-matched-v1/dense-device-check/dense',
        'planner-48-matched':ROOT/'hellaswag-matched-v1/evaluation/planner-50',
        'planner-40-matched':ROOT/'hellaswag-matched-v1/evaluation/planner-40'})
    return refs


def report():
    read = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
    results = [];comparisons = {};references={}
    for budget in TARGETS:
        p = OUT/'evaluation'/f'planner-{budget}.json';v=json.loads(p.read_text())
        v['target_retention']=TARGETS[budget]
        v['within_match_tolerance']=abs(v['suffix_kept']-TARGETS[budget])<=MATCH_TOLERANCE
        results.append(v)
        first=read(p.with_name(f'planner-{budget}-examples.jsonl'))
        for name,base in references_for(budget).items():
            ref=json.loads(base.with_suffix('.json').read_text())
            assert ref['corpus_sha256']==v['corpus_sha256'] and ref['examples']==v['examples']==10042
            references[name]=ref
            comparisons[f'planner-{budget}-vs-{name}']=paired(first,read(base.with_name(base.name+'-examples.jsonl')))
    if set(TARGETS)=={30,20}:
        comparisons['planner-20-vs-planner-30']=paired(read(OUT/'evaluation/planner-20-examples.jsonl'),read(OUT/'evaluation/planner-30-examples.jsonl'))
    atomic_json(OUT/'results.json',dict(status='complete' if all(x['within_match_tolerance'] for x in results) else 'retention_mismatch',
        results=results,references=references,paired=comparisons,match_tolerance=MATCH_TOLERANCE,
        reference_budget_note='Reused GRIFFIN/NeuroPrefetcher rows remain at their original approximately 48%/40% retention; not new 30%/20% baselines',
        calibration='1024 label-free train examples; thresholds frozen before full validation',
        protocol='dense prompts, first answer likelihood dense, subsequent answer likelihoods sparse; attention and first two MLP layers dense',
        griffin_comparison='generation-style adaptation, not reproduction of final-token-only classification mode'))
    assert all(x['within_match_tolerance'] for x in results),'Full-set retention drift exceeds 0.5pp; do not claim matched results'


def run(deadline):
    launch=OUT/'launch';launch.mkdir(parents=True,exist_ok=True);started=time.time();status='failed';stage='prepare'
    jobs=[('prepare',['prepare'])]
    for b in TARGETS:
        jobs += [(f'calibrate-{b}',['calibrate','--budget',str(b)]),
                 (f'evaluate-{b}',None)]
    jobs += [('report',['report'])]
    try:
        for stage,args in jobs:
            atomic_json(launch/'progress.json',dict(pid=os.getpid(),stage=stage,started_epoch=time.time(),deadline_epoch=deadline))
            cmd=[sys.executable,'-u','-m','scripts.mistral_hellaswag_matched']+args if args else [
                sys.executable,'-u','-m','scripts.mistral_hellaswag','--kind','planner','--budget',stage.split('-')[1],
                '--calibration',str(OUT/f"calibration-{stage.split('-')[1]}.json"),'--output-dir',str(OUT/'evaluation')]
            if args and set(TARGETS)=={30,20}:cmd+=['--lower']
            with (launch/(stage+'.log')).open('a') as log:
                p=subprocess.Popen(cmd,stdout=log,stderr=log)
                try:code=p.wait(timeout=max(1,deadline-time.time()-90))
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    raise TimeoutError('Matched-run spending deadline')
            atomic_json(launch/(stage+'-status.json'),dict(returncode=code,ended_epoch=time.time()))
            if code:raise RuntimeError(f'{stage} failed ({code})')
        status='complete'
    finally:atomic_json(launch/'completion.json',dict(status=status,last_stage=stage,started_epoch=started,ended_epoch=time.time()))


def main():
    global OUT,TARGETS
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','calibrate','report','run'])
    ap.add_argument('--budget',type=int,choices=[50,40,30,20]);ap.add_argument('--deadline-epoch',type=float)
    ap.add_argument('--lower',action='store_true',help='Evaluate final 30%/20% checkpoints in isolated outputs')
    a=ap.parse_args();OUT=ROOT/('hellaswag-lower-v1' if a.lower else 'hellaswag-matched-v1')
    TARGETS={30:.30,20:.20} if a.lower else {50:.48,40:.40}
    if a.stage=='calibrate':assert a.budget in TARGETS,'Budget does not belong to selected checkpoint group'
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.stage=='prepare':prepare()
    elif a.stage=='calibrate':calibrate(a.budget)
    elif a.stage=='report':report()
    else:
        assert a.deadline_epoch and 0<a.deadline_epoch-time.time()<=24*3600
        run(a.deadline_epoch)

if __name__=='__main__':main()
