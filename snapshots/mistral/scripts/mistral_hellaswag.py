"""Zero-shot HellaSwag with independent candidate histories and standard acc_norm."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from pathpred.collection import atomic_json,file_sha256
from scripts.mistral_comparisons import load,apply_calibration,ROOT,heartbeat
from scripts.mistral_comparison_core import score_batch

def choice_metrics(scores,choices,gold):
    assert len(scores)==len(choices)==4 and all(choices)
    raw=int(np.argmax(scores));norm=int(np.argmax(np.asarray(scores)/np.asarray([len(x) for x in choices])))
    return dict(prediction=raw,prediction_normalized=norm,accuracy=int(raw==gold),accuracy_normalized=int(norm==gold))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['dense','planner','neuro','griffin'],required=True)
    ap.add_argument('--budget',type=int,choices=[50,40,30,20],default=50);ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--calibration',type=Path);ap.add_argument('--output-dir',type=Path)
    a=ap.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.budget in (30,20):
        assert a.kind=='planner' and a.calibration and a.output_dir,'Lower checkpoints require explicit calibration and isolated outputs'
    label=a.kind if a.kind=='dense' else f'{a.kind}-{a.budget}'
    root=a.output_dir or ROOT/('hellaswag-smoke' if a.limit else 'hellaswag');root.mkdir(parents=True,exist_ok=True)
    final=root/(label+'.json')
    corpus=Path('data/hellaswag-mistral-v1/examples.jsonl')
    provenance=None
    if a.calibration:
        assert a.kind=='planner' and a.output_dir,'Custom calibration needs isolated planner outputs'
        calibration=json.loads(a.calibration.read_text());assert calibration['checkpoint_budget']==a.budget
        provenance=dict(calibration_sha256=file_sha256(a.calibration),corpus_sha256=file_sha256(corpus),
            code={p:file_sha256(Path(p)) for p in ['scripts/mistral_hellaswag.py','scripts/mistral_comparison_core.py','scripts/mistral_comparisons.py']},
            limit=a.limit,budget=a.budget)
        identity=root/(label+'-identity.json')
        if identity.exists():assert json.loads(identity.read_text())==provenance,'Refuse mixed evaluation provenance'
        else:
            assert not final.exists() and not (root/(label+'-examples.jsonl')).exists()
            atomic_json(identity,provenance)
    if final.exists():return
    examples=[json.loads(x) for x in corpus.read_text().splitlines()]
    if a.limit:examples=examples[:a.limit]
    else:assert len(examples)==10042
    pol=load(a.kind,a.budget)
    if a.calibration:
        from scripts.mistral_hellaswag_matched import checkpoint_identity
        assert calibration['checkpoint']==checkpoint_identity(a.budget),'Checkpoint changed since calibration'
        pol.offsets=torch.tensor(calibration['offsets'],device='cuda');pol.keep=calibration['target']
    else:apply_calibration(pol,a.budget,64)
    path=root/(label+'-examples.jsonl');done={}
    if path.exists():
        for line in path.read_text().splitlines():
            r=json.loads(line);assert r['id'] not in done;done[r['id']]=r
    pending=sorted([x for x in examples if x['id'] not in done],key=lambda x:(max(map(len,x['sequences'])),x['id']))
    started=time.time()
    # Four examples = sixteen independent candidates. No candidate can change another's cache/path.
    for n in range(0,len(pending),4):
        batch=pending[n:n+4];seq=[s for x in batch for s in x['sequences']];prompts=[p for x in batch for p in x['prompt_lengths']]
        z=score_batch(pol,seq,prompts,teacher=False)
        with path.open('a') as f:
            for i,x in enumerate(batch):
                scores=(-z['sums'][i*4:i*4+4,1]).tolist();r=dict(id=x['id'],gold=x['gold'],scores=scores,
                    kept_sum=float(z['kept_sum'][i*4:i*4+4].sum()),executions=float(z['executions'][i*4:i*4+4].sum()),
                    **choice_metrics(scores,x['choices'],x['gold']))
                f.write(json.dumps(r)+'\n');f.flush();done[r['id']]=r
        progress=dict(method=label,complete=len(done),total=len(examples),seconds_this_process=time.time()-started)
        atomic_json(root/(label+'-progress.json'),progress);print(json.dumps(progress),flush=True)
    values=[done[x['id']] for x in examples];N=len(values)
    atomic_json(final,dict(method=label,examples=N,accuracy=sum(x['accuracy'] for x in values)/N,
        accuracy_normalized=sum(x['accuracy_normalized'] for x in values)/N,
        suffix_kept=1.0 if a.kind=='dense' else sum(x['kept_sum'] for x in values)/max(1,sum(x['executions'] for x in values)),
        zero_shot=True,add_bos_token=True,prompt_fully_dense=True,dense_prefix=2,
        candidate_histories_independent=True,normalization='character count of preprocessed ending',
        corpus_sha256=file_sha256(corpus),full_validation=not bool(a.limit),
        protocol='dense prompt; first answer likelihood from dense prompt logits; subsequent answer likelihoods from sparse decoding',
        attention_dense=True,final_answer_input_not_executed=True,
        retention_denominator='eligible MLP neurons averaged over executed answer inputs, layers 2-31; all four candidates; padding excluded',
        calibration_provenance=provenance))
if __name__=='__main__':main()
