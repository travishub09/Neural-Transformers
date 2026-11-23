"""Explicit frozen-run inheritance; never rewrite a parent's producing records."""
import hashlib,shutil
from pathlib import Path
from scripts.followup_common import read,write

def copy_records(source,dest):
    files={}
    for p in source.rglob('*'):
        if not p.is_file() or (p.suffix not in ('.json','.jsonl','.safetensors') and p.name!='last.pt'):continue
        relative=p.relative_to(source);target=dest/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():continue
        if p.suffix=='.safetensors' or p.name=='last.pt':target.symlink_to(p.resolve())
        else:shutil.copy2(p,target)
        files[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return files

def inherit_qwen(c,root,out):
    source=Path(c['resume_artifact_source']);assert source.exists(),str(source)
    old=Path(c['resume_collection'])/'raw';proof={}
    for stage in ['copy','scheduled','budget-50','budget-40']:
        src=source/'autoregressive'/stage
        assert read(src/'completion.json')['status']=='complete'
        for p in (old/'autoregressive'/stage).glob('checkpoint-*.json'):
            meta=read(p);weight=src/meta['file']['file']
            assert hashlib.sha256(weight.read_bytes()).hexdigest()==meta['file']['sha256']
        for name in ['training.json','completion.json','history.jsonl']:
            assert hashlib.sha256((src/name).read_bytes()).digest()==hashlib.sha256((old/'autoregressive'/stage/name).read_bytes()).digest()
        proof[stage]=copy_records(src,out/'autoregressive'/stage)
    if c.get('resume_partial_one_shot'):
        import torch
        src=source/'one_shot_history'/'copy';state=torch.load(src/'last.pt',map_location='cpu',weights_only=True)
        assert state['tokens']==c['resume_partial_tokens'] and 'optimizer' in state and 'cpu_rng' in state and 'cuda_rng' in state
        proof['one_shot_copy_partial']=copy_records(src,out/'one_shot_history'/'copy')
        parent_identity=Path(c['resume_collection'])/'attempts/interrupted-20261123/identity.json'
        if parent_identity.exists():write(root/'partial-checkpoint-producing-identity.json',read(parent_identity))
        history=out/'one_shot_history'/'copy'/'history.jsonl';rows=history.read_text().splitlines();tail=[r for r in rows if read_line_tokens(r)>state['tokens']]
        if tail:
            (root/'uncommitted-training-tail.jsonl').write_text('\n'.join(tail)+'\n')
            history.write_text('\n'.join(r for r in rows if read_line_tokens(r)<=state['tokens'])+'\n')
    write(root/'resume-lineage.json',dict(source_collection=c['resume_collection'],source_artifacts=str(source),inherited_variant='autoregressive',stages=proof,optimizer_rng='Completed stages reused unchanged; partial one-shot restores saved optimizer and CPU/CUDA RNG when configured.',producing_identity=read(Path(c['resume_collection'])/'identity.json')))

def read_line_tokens(line):
    import json
    return json.loads(line)['tokens']

def inherit_reasoning(c,root,out):
    source=Path(c['resume_collection']);manifest=read(source/'source-data-manifest.json')
    assert manifest==read(root/'source-data-manifest.json')
    proof={}
    for p in (source/'raw').glob('*-examples.json'):
        target=out/p.name
        if not target.exists():shutil.copy2(p,target)
        proof[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    write(root/'resume-lineage.json',dict(source_collection=str(source),source_files=proof,producing_identity=read(source/'identity.json'),method='Reuse completed per-example records; run only missing arms/examples with unchanged batches, prompts, weights and thresholds.'))
