"""Pinned, inference-only loading and record helpers for the follow-up suite."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from pathpred.collection import atomic_json, file_sha256
from pathpred.gated_qwen import gate_mistral
from pathpred.open_loop import SuffixPlanner, PlannerConfig
from scripts.mistral_comparison_core import Policy


def utc(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def read(path): return json.loads(Path(path).read_text())
def write(path,value): atomic_json(Path(path),value)


def backbone():
    c=read('configs/mistral-lower-budgets.json')
    model=AutoModelForCausalLM.from_pretrained(c['model_path'],torch_dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='eager',local_files_only=True).eval()
    return model,gate_mistral(model)


def planner_policy(model,ctrl,budget):
    family='mistral-lower-v1' if budget in (20,30) else 'mistral-phased-v1'
    root=Path('artifacts')/family/f'budget-{budget}'
    done=read(root/'completion.json'); n=done['tokens']
    meta=read(root/f'checkpoint-{n:08d}.json'); path=root/meta['file']['file']
    assert file_sha256(path)==meta['file']['sha256']
    p=SuffixPlanner(PlannerConfig(llm_layers=32,units=14336,hidden_size=4096)).cuda()
    p.load_state_dict(load_file(str(path),device='cuda'));p.eval()
    qpath=Path('artifacts/mistral-pilot-v1/thresholds.safetensors')
    q=load_file(str(qpath),device='cuda')['q75']
    state=read(root/f'check-{n:08d}/validation.json')
    pol=Policy(model,ctrl,'planner',planner=p,q=q,
        offsets=torch.tensor(state['offsets'],device='cuda'),keep=budget/100)
    provenance=dict(checkpoint=str(path),sha256=meta['file']['sha256'],metadata=meta,
        thresholds_sha256=file_sha256(qpath),validation=state,
        completion=done,lineage=read(root/'lineage.json') if (root/'lineage.json').exists() else 'see checkpoint metadata')
    return pol,provenance


def source_rows():
    root=Path('data/mistral-long-prompt-v1');manifest=read(root/'manifest.json')
    rows={}
    for split in ['validation','test']:
        path=root/f'{split}.jsonl'
        assert file_sha256(path)==manifest['files'][path.name]
        rows[split]=[json.loads(x) for x in path.read_text().splitlines()]
    assert not {x['document_sha256'] for x in rows['validation']} & {x['document_sha256'] for x in rows['test']}
    return rows,manifest


def start_record(collection,argv):
    root=Path(collection);root.mkdir(parents=True,exist_ok=True)
    git=lambda *args: subprocess.check_output(['git',*args],text=True).strip()
    dirty=bool(git('status','--porcelain','--untracked-files=no'))
    identity=dict(collection=root.name,run='evaluation',start_utc=utc(),end_utc=None,
        git_commit=git('rev-parse','HEAD'),branch=git('branch','--show-current'),dirty_tree=dirty)
    write(root/'identity.json',identity)
    if dirty:(root/'code.diff').write_text(git('diff','HEAD'))
    write(root/'command.json',dict(argv=argv))
    write(root/'hardware.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cuda=torch.version.cuda,python=sys.version,threads=torch.get_num_threads()))
    return identity


def fingerprint(root):
    files={str(p):file_sha256(p) for d in ['scripts','pathpred'] for p in Path(d).rglob('*.py')}
    write(Path(root)/'fingerprints.json',files)
    (Path(root)/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))


def ci(rows,stat,seed=20261122,repetitions=2000):
    rng=np.random.default_rng(seed)
    draws=[stat([rows[i] for i in rng.integers(len(rows),size=len(rows))]) for _ in range(repetitions)]
    return np.quantile(draws,[.025,.975]).tolist()
