"""Long-prompt continuation perplexity with separate retention calibration."""
import argparse
from collections import Counter
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
from scripts.mistral_comparisons import load, score_rows, paired
from scripts.oracle_quality import aggregate, bootstrap_quality
from scripts.mistral_hellaswag_matched import checkpoint_identity

ROOT=Path('artifacts/mistral-comparisons-v1/long-prompt-v1')
DATA=Path('data/mistral-long-prompt-v1')
CONFIG=Path('configs/mistral-long-prompt.json')


def config():return json.loads(CONFIG.read_text())


def ranked_indices(n, count, split, seed):
    if n<count:raise ValueError(f'Only {n} complete long blocks available; need {count}')
    return sorted(range(n),key=lambda i:hashlib.sha256(f'{seed}:{split}:{i}'.encode()).hexdigest())[:count]


def crop(tokens, prompt, continuation=512):
    if len(tokens)!=1536 or prompt not in (512,1024) or continuation!=512:
        raise ValueError('Expected 1536-token block and supported context/target lengths')
    return tokens[1024-prompt:1536]


def cap_retention(hist, cap, units=14336):
    if not hist or sum(hist.values())<=0:raise ValueError('Empty count histogram')
    return sum(min(int(n),cap)*int(count) for n,count in hist.items())/(sum(hist.values())*units)


def matched_cap(hist, target, max_cap=7168):
    # Layer-zero released predictor features are unaffected by higher-layer masks.
    # Thus reducing the cap clips each native count exactly, without another rollout.
    lo,hi=1,max_cap
    while hi-lo>1:
        mid=(lo+hi)//2
        if cap_retention(hist,mid)<target:lo=mid
        else:hi=mid
    return min((lo,hi),key=lambda x:abs(cap_retention(hist,x)-target))


def prepare():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    c=config();DATA.mkdir(parents=True,exist_ok=True)
    manifest=DATA/'manifest.json'
    if manifest.exists():
        value=json.loads(manifest.read_text());assert value['config_sha256']==file_sha256(CONFIG)
        for name,sha in value['files'].items():assert file_sha256(DATA/name)==sha
        atomic_json(ROOT/'data-manifest.json',value);return
    pin=DATA/'source.json'
    if pin.exists():source=json.loads(pin.read_text())
    else:
        source=dict(dataset=c['dataset'],config=c['dataset_config'],revision=HfApi().dataset_info(c['dataset']).sha)
        atomic_json(pin,source)
    tok=AutoTokenizer.from_pretrained('checkpoints/mistral-7b-v0.1',local_files_only=True)
    meta={};hashes=set()
    for split,count in [(c['calibration_split'],c['calibration_windows']),(c['test_split'],c['test_windows'])]:
        ds=load_dataset(source['dataset'],source['config'],revision=source['revision'],split=split)
        text='\n\n'.join(row['text'] for row in ds if row['text'].strip())
        tokens=tok.encode(text,add_special_tokens=False)
        indices=ranked_indices(len(tokens)//c['window_tokens'],count,split,c['seed'])
        rows=[]
        for i in indices:
            seq=tokens[i*1536:(i+1)*1536]
            sha=hashlib.sha256(json.dumps(seq,separators=(',',':')).encode()).hexdigest()
            assert sha not in hashes,'Duplicated calibration/test block';hashes.add(sha)
            rows.append(dict(document_sha256=sha,domain='wikitext',split=split,block_index=i,
                start_token=i*1536,token_ids=seq))
        dest=DATA/f'{split}.jsonl';tmp=dest.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(x)+'\n' for x in rows));tmp.replace(dest)
        meta[split]=dict(total_tokens=len(tokens),complete_blocks=len(tokens)//1536,selected_blocks=indices,
            selected_sha256=[r['document_sha256'] for r in rows],text_sha256=hashlib.sha256(text.encode()).hexdigest())
    value=dict(source=source,config=c,config_sha256=file_sha256(CONFIG),splits=meta,
        tokenizer_model='mistralai/Mistral-7B-v0.1',tokenizer_revision='27d67f1b5f57dc0953326b2601d68371d40ea8da',
        files={p.name:file_sha256(p) for p in DATA.glob('*.jsonl')},
        scope='Official validation for retention calibration; official test for final evaluation. Base-model pretraining contamination is not assessed.',
        inference_weights_frozen=True)
    atomic_json(manifest,value);atomic_json(ROOT/'data-manifest.json',value)


def rows(split,prompt):
    c=config();manifest=json.loads((DATA/'manifest.json').read_text());p=DATA/f'{split}.jsonl'
    assert file_sha256(p)==manifest['files'][p.name]
    values=[json.loads(x) for x in p.read_text().splitlines()]
    return [(dict(document_sha256=x['document_sha256'],domain=x['domain']),crop(x['token_ids'],prompt)) for x in values]


def identity():
    files=[CONFIG,Path(__file__),Path('scripts/mistral_comparison_core.py'),Path('scripts/mistral_comparisons.py'),
        Path('scripts/oracle_quality.py'),Path('scripts/mistral_hellaswag_matched.py')]+sorted(Path('pathpred').glob('*.py'))
    value=dict(code={str(p):file_sha256(p) for p in files},data=file_sha256(DATA/'manifest.json'),
        checkpoints={str(b):checkpoint_identity(b) for b in (50,40)})
    assets=Path('/workspace/neuroprefetcher-reference/assets')
    value['released_assets']={p.name:file_sha256(p) for p in assets.glob('*.pt')}
    value['released_engine_commit']=subprocess.check_output(['git','-C',str(assets.parent),'rev-parse','HEAD'],text=True).strip()
    value['device']=torch.cuda.get_device_name();value['torch']=torch.__version__
    path=ROOT/'identity.json'
    if path.exists():assert json.loads(path.read_text())==value,'Refuse mixed long-prompt provenance'
    else:atomic_json(path,value)


def measure(pol,items,prompt,quality=False):
    return score_rows(pol,items,prompt,quality=quality,batch_size=config()['batch_size'])


def calibrate_neuro(prompt):
    c=config();out=ROOT/f'prompt-{prompt}';out.mkdir(parents=True,exist_ok=True)
    if all((out/f'neuro-{b}-calibration.json').exists() for b in (50,40)):return
    hist_path=out/'neuro-native-histogram.json'
    if hist_path.exists():hist=json.loads(hist_path.read_text())
    else:
        pol=load('neuro',50);pol.neuro.cap=7168;hist=Counter();original=pol.masks
        def capture():
            masks=original()
            if masks is not None:
                hist.update(masks.sum(-1).cpu().flatten().tolist())
            return masks
        pol.masks=capture
        native,_=measure(pol,rows(c['calibration_split'],prompt),prompt)
        assert sum(hist.values())==c['calibration_windows']*511*30
        assert abs(native-cap_retention(hist,7168))<1e-10
        atomic_json(hist_path,dict(hist));del pol;torch.cuda.empty_cache()
    audit_path=out/'neuro-cap-verification.json'
    if not audit_path.exists():
        cap=matched_cap(hist,c['targets']['40']);expected=cap_retention(hist,cap)
        pol=load('neuro',40);pol.neuro.cap=cap
        observed,_=measure(pol,rows(c['calibration_split'],prompt),prompt)
        assert abs(observed-expected)<1e-8,'Native-count clipping does not reproduce real capped rollout'
        atomic_json(audit_path,dict(cap=cap,expected_retention=expected,observed_retention=observed,
            passed=True,scope='Full calibration set, same prompt length, real capped causal rollout'))
        del pol;torch.cuda.empty_cache()
    for b in (50,40):
        target=c['targets'][str(b)];cap=matched_cap(hist,target);used=cap_retention(hist,cap)
        assert abs(used-target)<=c['calibration_tolerance'],'Released predictor cannot reach requested retention'
        atomic_json(out/f'neuro-{b}-calibration.json',dict(cap=cap,target=target,calibration_retention=used,
            split=c['calibration_split'],windows=c['calibration_windows'],prompt_tokens=prompt,
            adaptation='Retention-only calibrated active-index cap; released predictor and centroid bank unchanged',
            calibration_method='Exact expected capped retention from native per-layer count histogram; layer-zero features independent of cap'))


def calibrate_planner(prompt,budget):
    c=config();out=ROOT/f'prompt-{prompt}';out.mkdir(parents=True,exist_ok=True)
    path=out/f'planner-{budget}-calibration.json'
    if path.exists():return
    pol=load('planner',budget);base=pol.offsets.clone();items=rows(c['calibration_split'],prompt)
    trials_path=out/f'planner-{budget}-trials.json';trials=json.loads(trials_path.read_text()) if trials_path.exists() else {}
    target=c['targets'][str(budget)]
    def evaluate(shift):
        key=str(shift)
        if key in trials:return trials[key]
        pol.offsets=base+shift
        used,_=measure(pol,items,prompt)
        trials[key]=used;atomic_json(trials_path,trials)
        print(json.dumps(dict(stage='calibration',prompt=prompt,budget=budget,shift=shift,retention=used)),flush=True)
        return used
    result=calibrate_threshold(evaluate,target,tolerance=c['calibration_tolerance'],max_evaluations=18)
    atomic_json(path,dict(target=target,offsets=(base+result['shift']).cpu().tolist(),calibration=result,
        calibration_retention=result['validation_retention'],split=c['calibration_split'],windows=len(items),
        prompt_tokens=prompt,checkpoint_budget=budget,objective='Retention only; no perplexity/KL/accuracy selection'))


def evaluate(kind,prompt,budget):
    c=config();out=ROOT/f'prompt-{prompt}';out.mkdir(parents=True,exist_ok=True)
    label='dense' if kind=='dense' else f'{kind}-{budget}';path=out/f'{label}.json'
    if path.exists():return
    target=c['targets'][str(budget)]
    pol=load(kind,budget);pol.keep=target
    if kind in ('planner','neuro'):
        cal=json.loads((out/f'{kind}-{budget}-calibration.json').read_text())
        if kind=='planner':pol.offsets=torch.tensor(cal['offsets'],device='cuda')
        else:pol.neuro.cap=cal['cap']
    items=rows(c['test_split'],prompt);tick=time.time()
    kept,docs=measure(pol,items,prompt,quality=True)
    assert len(docs)==c['test_windows'] and all(x['tokens']==512 and (kind=='dense' or x['sparse_executions']==511) for x in docs)
    overall=aggregate(docs);overall.update(bootstrap_quality(docs,c['bootstrap_repetitions'],c['seed']))
    overall['routed_kept']=1. if kind=='dense' else kept
    atomic_json(out/f'{label}-documents.json',docs)
    record=dict(method=kind,checkpoint_budget=budget if kind=='planner' else None,prompt_tokens=prompt,
        continuation_tokens=512,windows=len(items),overall=overall,seconds=time.time()-tick,
        target_retention=None if kind=='dense' else target,
        within_match_tolerance=kind=='dense' or abs(kept-target)<=c['test_retention_tolerance'],
        test_retention_tolerance=c['test_retention_tolerance'],protocol=c['protocol'],
        context_and_targets='Both prompt lengths score the identical final 512 tokens of each 1536-token block',
        retention_denominator='Eligible layers 2-31; 511 executed continuation inputs per window; prompt excluded',
        data_manifest_sha256=file_sha256(DATA/'manifest.json'),identity_sha256=file_sha256(ROOT/'identity.json'))
    atomic_json(path,record)


def report():
    c=config();results=[];comparisons={}
    for prompt in c['prompt_tokens']:
        out=ROOT/f'prompt-{prompt}'
        dense=json.loads((out/'dense-documents.json').read_text());dense_by={d['document']:d for d in dense}
        for b in (50,40):
            for kind in ('planner','neuro','griffin'):
                p=out/f'{kind}-{b}.json';record=json.loads(p.read_text());results.append(record)
                docs=json.loads((out/f'{kind}-{b}-documents.json').read_text())
                assert all(abs(d['teacher_nll']-dense_by[d['document']]['teacher_nll'])<.01 for d in docs)
            a=json.loads((out/f'planner-{b}-documents.json').read_text())
            for kind in ('neuro','griffin'):
                comparisons[f'prompt-{prompt}-budget-{b}-vs-{kind}']=paired(a,json.loads((out/f'{kind}-{b}-documents.json').read_text()),seed=c['seed'])
    matched=all(x['within_match_tolerance'] for x in results)
    atomic_json(ROOT/'results.json',dict(status='complete' if matched else 'retention_mismatch',results=results,
        paired=comparisons,calibration='Separate official WikiText validation split; no test quality used',
        limitation='Bootstrap units are nonoverlapping blocks; adjacent blocks can share an article, so intervals are descriptive, not article-clustered.'))
    # Preserve every row even if retention drifts; never silently call it matched.
    if not matched:raise RuntimeError('One or more test retentions miss the predeclared tolerance; review results.json')


def jobs():
    yield 'prepare',['prepare']
    yield 'identity',['identity']
    for prompt in config()['prompt_tokens']:
        flags=['--prompt',str(prompt)]
        yield f'dense-{prompt}',['evaluate','--kind','dense']+flags
        yield f'cal-neuro-{prompt}',['calibrate-neuro']+flags
        for b in (50,40):
            bf=flags+['--budget',str(b)]
            yield f'cal-planner-{prompt}-{b}',['calibrate-planner']+bf
            for kind in ('griffin','neuro','planner'):
                yield f'eval-{prompt}-{b}-{kind}',['evaluate','--kind',kind]+bf
    yield 'report',['report']


def run(deadline):
    launch=ROOT/'launch';launch.mkdir(parents=True,exist_ok=True);started=time.time();status='failed';stage='starting'
    try:
        for stage,arguments in jobs():
            atomic_json(launch/'progress.json',dict(pid=os.getpid(),stage=stage,started_epoch=time.time(),deadline_epoch=deadline))
            with (launch/(stage+'.log')).open('a') as log:
                p=subprocess.Popen([sys.executable,'-u','-m','scripts.mistral_long_prompt']+arguments,stdout=log,stderr=log)
                try:code=p.wait(timeout=max(1,deadline-time.time()-90))
                except subprocess.TimeoutExpired:
                    p.terminate()
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    raise TimeoutError('Long-prompt spending deadline')
            atomic_json(launch/(stage+'-status.json'),dict(returncode=code,ended_epoch=time.time()))
            if code:raise RuntimeError(f'{stage} failed ({code})')
        status='complete'
    finally:atomic_json(launch/'completion.json',dict(status=status,last_stage=stage,started_epoch=started,ended_epoch=time.time()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','identity','calibrate-neuro','calibrate-planner','evaluate','report','run'])
    ap.add_argument('--prompt',type=int,choices=[512,1024]);ap.add_argument('--budget',type=int,choices=[50,40],default=50)
    ap.add_argument('--kind',choices=['planner','neuro','griffin','dense']);ap.add_argument('--deadline-epoch',type=float)
    a=ap.parse_args();ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.stage=='prepare':prepare()
    elif a.stage=='identity':identity()
    elif a.stage=='calibrate-neuro':calibrate_neuro(a.prompt)
    elif a.stage=='calibrate-planner':calibrate_planner(a.prompt,a.budget)
    elif a.stage=='evaluate':evaluate(a.kind,a.prompt,a.budget)
    elif a.stage=='report':report()
    else:
        assert a.deadline_epoch and 0<a.deadline_epoch-time.time()<=24*3600
        run(a.deadline_epoch)

if __name__=='__main__':main()
