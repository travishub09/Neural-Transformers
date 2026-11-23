"""Identical WikiText targets under different prompt lengths; log every loss."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from scripts.followup_common import backbone,planner_policy,source_rows,start_record,fingerprint,read,write,utc,ci
from scripts.followup_speculative import quality_summary
from scripts.mistral_comparison_core import score_batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/followup-positions.json');a=p.parse_args();c=read(a.config)
    root=Path(c['collection']);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False
    identity=start_record(root,sys.argv);fingerprint(root)
    try:
        sources,manifest=source_rows();test=sources['test'];assert len(test)==64
        write(root/'data-manifest.json',manifest);model,ctrl=backbone();summary={}
        for budget in c['budgets']:
            pol,prov=planner_policy(model,ctrl,budget);write(root/f'checkpoint-{budget}.json',prov)
            for prompt,calibration_prompt in c['prompt_and_calibration']:
                cal=read(f'artifacts/mistral-comparisons-v1/long-prompt-v1/prompt-{calibration_prompt}/planner-{budget}-calibration.json')
                pol.offsets=torch.tensor(cal['offsets'],device='cuda');pol.keep=cal['target']
                name=f'planner-{budget}-prompt-{prompt}-cal-{calibration_prompt}'
                write(root/f'{name}-calibration.json',cal);docs=[];position_rows=[]
                for begin in range(0,64,c['batch_size']):
                    batch=test[begin:begin+c['batch_size']];captured=[]
                    def capture(t,vals,live,ret):
                        # NaN is never serialized: dense first prediction uses retention 1.
                        captured.append(torch.cat([vals,torch.ones_like(vals[:,:1]) if ret is None else ret[:,None]],1))
                    z=score_batch(pol,[r['token_ids'][1024-prompt:1536] for r in batch],[prompt]*len(batch),position_callback=capture)
                    values=torch.stack(captured,1).cpu().tolist()
                    for i,row in enumerate(batch):
                        d=dict(zip(['teacher_nll','student_nll','kl','top1_agreement','tokens'],z['sums'][i].tolist()))
                        d.update(document=row['document_sha256'],domain='wikitext',kept_sum=float(z['kept_sum'][i]),executions=float(z['executions'][i]))
                        docs.append(d);position_rows.append(dict(document=row['document_sha256'],metrics=values[i]))
                    write(out/f'{name}-documents.json',docs)
                    write(out/f'{name}-positions.json',dict(columns=['dense_nll','student_nll','kl','top1_agreement','scored','retention'],
                        target_positions=list(range(prompt,prompt+512)),source_target_indices=list(range(1024,1536)),
                        note='Position zero in continuation is predicted by dense prompt; retention at that prediction is 1 and excluded from sparse-retention denominator.',documents=position_rows))
                    print(json.dumps(dict(condition=name,documents=len(docs))),flush=True)
                v=quality_summary(docs)
                for metric in ['relative_ppl_increase','teacher_kl_nats','top1_agreement','routed_kept','dense_perplexity','student_perplexity']:
                    v[metric+'_ci95']=ci(docs,lambda rs:quality_summary(rs)[metric])
                arr=np.array([r['metrics'] for r in position_rows]);bins=[]
                for start in range(0,512,32):
                    a=arr[:,start:start+32];lossgap=(a[:,:,1]-a[:,:,0]).mean(1)
                    bins.append(dict(continuation_start=start,continuation_end_exclusive=start+32,
                        target_position_start=prompt+start,dense_nll=float(a[:,:,0].mean()),student_nll=float(a[:,:,1].mean()),
                        excess_nll=float(lossgap.mean()),excess_nll_ci95=ci(lossgap.tolist(),lambda rs:float(np.mean(rs))),
                        kl=float(a[:,:,2].mean()),top1_agreement=float(a[:,:,3].mean()),
                        retention=float(arr[:,max(1,start):start+32,5].mean())))
                v.update(bins=bins,prompt_tokens=prompt,calibration_prompt_tokens=calibration_prompt,scored_tokens=32768,
                    single_seed=True,bootstrap='paired nonoverlapping WikiText block percentile bootstrap; not independent original documents or seed variation')
                summary[name]=v;write(out/'summary.json',summary)
            del pol;torch.cuda.empty_cache()
        paired={}
        for budget in c['budgets']:
            for prompt in [512,1024]:
                key=lambda p:f'planner-{budget}-prompt-{p}-cal-512'
                aa=read(out/f'{key(64)}-positions.json')['documents'];bb=read(out/f'{key(prompt)}-positions.json')['documents']
                assert [x['document'] for x in aa]==[x['document'] for x in bb]
                values=[]
                for x,y in zip(aa,bb):
                    x,y=np.array(x['metrics']),np.array(y['metrics'])
                    values.append(dict(excess_nll_difference=float(((y[:,1]-y[:,0])-(x[:,1]-x[:,0])).mean()),
                        dense_nll_difference=float((y[:,0]-x[:,0]).mean()),retention_difference=float((y[1:,5]-x[1:,5]).mean())))
                paired[f'{budget}-{prompt}-minus-64']={k:dict(mean=float(np.mean([r[k] for r in values])),
                    ci95=ci(values,lambda rs:float(np.mean([r[k] for r in rs])))) for k in values[0]}
        write(out/'paired.json',paired)
        write(root/'compute.json',dict(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),training_tokens=0,throughput_kind='quality evaluation, not inference speed'))
        write(root/'completion.json',dict(status='complete',finished_utc=utc(),reason='All fixed conditions executed; review pending.'))
    except BaseException as e:
        write(root/'completion.json',dict(status='failed',finished_utc=utc(),reason=repr(e)));raise
    finally:identity['end_utc']=utc();write(root/'identity.json',identity);fingerprint(root)


if __name__=='__main__':main()
