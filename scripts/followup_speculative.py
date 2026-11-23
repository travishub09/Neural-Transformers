"""Preregistered batch-one greedy self-speculation and separate quality audit."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch
from scripts.followup_common import (backbone,planner_policy,source_rows,start_record,
    fingerprint,read,write,utc,ci)
from scripts.mistral_comparison_core import Policy,score_batch
from pathpred.speculative import dense_greedy,speculative_greedy


def quality_summary(rows):
    import math
    n=sum(x['tokens'] for x in rows)
    t,s,k,a=[sum(x[key] for x in rows)/n for key in ['teacher_nll','student_nll','kl','top1_agreement']]
    return dict(dense_perplexity=math.exp(t),student_perplexity=math.exp(s),
        relative_ppl_increase=math.expm1(s-t),teacher_kl_nats=k,top1_agreement=a,
        routed_kept=sum(x['kept_sum'] for x in rows)/sum(x['executions'] for x in rows),tokens=n)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/followup-speculative.json');a=p.parse_args()
    c=read(a.config);root=Path(c['collection']);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False
    identity=start_record(root,sys.argv);fingerprint(root)
    try:
        rows,manifest=source_rows();write(root/'data-manifest.json',manifest)
        test=rows['test'][:c['documents']];warm=rows['validation'][0]
        write(root/'selected-data.json',{s:[{k:v for k,v in x.items() if k!='token_ids'} for x in rs]
            for s,rs in [('test',test),('warmup',[warm])]})
        model,ctrl=backbone();dense=Policy(model,ctrl,'dense')
        write(root/'actual-parameters.json',dict(backbone=sum(x.numel() for x in model.parameters()),
             by_parameter={k:v.numel() for k,v in model.named_parameters()}))
        dense_greedy(dense,warm['token_ids'][:c['prompt_tokens']],8)
        baseline=[]
        for row in test:
            z=dense_greedy(dense,row['token_ids'][:c['prompt_tokens']],c['new_tokens'])
            z.update(document=row['document_sha256'],domain='wikitext',routed_kept=1.)
            baseline.append(z);write(out/'dense.json',baseline)
        summary={'protocol':c['protocol'],'single_seed':True,'conditions':{}}
        for budget in c['budgets']:
            pol,provenance=planner_policy(model,ctrl,budget);write(root/f'checkpoint-{budget}.json',provenance)
            quality=[]
            # Separate teacher-forced audit: no test tuning and no timing claims.
            for begin in range(0,len(test),8):
                batch=test[begin:begin+8];length=c['prompt_tokens']+c['quality_tokens']
                z=score_batch(pol,[x['token_ids'][:length] for x in batch],[c['prompt_tokens']]*len(batch))
                for i,r in enumerate(batch):
                    q=dict(zip(['teacher_nll','student_nll','kl','top1_agreement','tokens'],z['sums'][i].tolist()))
                    q.update(document=r['document_sha256'],domain='wikitext',kept_sum=float(z['kept_sum'][i]),executions=float(z['executions'][i]))
                    quality.append(q)
                write(out/f'quality-{budget}-documents.json',quality)
            qs=quality_summary(quality)
            for metric in ['dense_perplexity','student_perplexity','relative_ppl_increase','teacher_kl_nats','top1_agreement','routed_kept']:
                qs[metric+'_ci95']=ci(quality,lambda rs:quality_summary(rs)[metric])
            qs['bootstrap']='percentile paired nonoverlapping WikiText block bootstrap; evaluation noise, not seed variation'
            write(out/f'quality-{budget}.json',qs)
            for block in c['blocks']:
                speculative_greedy(pol,warm['token_ids'][:c['prompt_tokens']],8,block)
                results=[]
                for row,base in zip(test,baseline):
                    z=speculative_greedy(pol,row['token_ids'][:c['prompt_tokens']],c['new_tokens'],block)
                    z.update(document=row['document_sha256'],domain='wikitext',
                        dense_decode_seconds=base['decode_seconds'],dense_prefill_seconds=base['prefill_seconds'],
                        exact_match=z['tokens']==base['tokens'])
                    results.append(z);write(out/f'planner-{budget}-k{block}-documents.json',results)
                    print(json.dumps(dict(budget=budget,block=block,documents=len(results),exact_match=z['exact_match'])),flush=True)
                def speed(rs):return sum(x['dense_decode_seconds'] for x in rs)/sum(x['decode_seconds'] for x in rs)
                def total_speed(rs):return sum(x['dense_decode_seconds']+x['dense_prefill_seconds'] for x in rs)/sum(x['decode_seconds']+x['prefill_seconds'] for x in rs)
                def accepted(rs):return sum(s['accepted'] for r in rs for s in r['steps'])/sum(len(r['steps']) for r in rs)
                nsteps=sum(len(r['steps']) for r in results);proposed=sum(s['proposed'] for r in results for s in r['steps'])
                v=dict(exact_match_all=all(x['exact_match'] for x in results),decode_speedup=speed(results),decode_speedup_ci95=ci(results,speed),
                    accepted_draft_tokens_per_verification=accepted(results),accepted_ci95=ci(results,accepted),
                    proposed_acceptance_fraction=sum(s['accepted'] for r in results for s in r['steps'])/proposed,
                    emitted_tokens_per_verification=sum(len(x['tokens']) for x in results)/nsteps,
                    total_speedup_including_prefill=total_speed(results),total_speedup_ci95=ci(results,total_speed),
                    inference_tokens_per_second=sum(len(x['tokens']) for x in results)/sum(x['decode_seconds'] for x in results),
                    draft_routed_kept=sum(x['draft_retention']*x['draft_sparse_executions'] for x in results)/sum(x['draft_sparse_executions'] for x in results),
                    teacher_forced_routed_kept=qs['routed_kept'],quality_source=f'quality-{budget}.json',
                    implementation='dense masked draft; shared backbone weights; separate KV; batch 1; synchronized wall clock; includes rollback, planner, retention accounting and host token selection',
                    bootstrap=qs['bootstrap'])
                v['speedup_supported']=v['exact_match_all'] and v['decode_speedup_ci95'][0]>1
                summary['conditions'][f'planner-{budget}-k{block}']=v
                write(out/'summary.json',summary)
            del pol;torch.cuda.empty_cache()
        write(root/'compute.json',dict(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            throughput_kind='inference; see per-condition summary',training_tokens=0))
        write(root/'completion.json',dict(status='complete',finished_utc=utc(),reason='All preregistered conditions executed; scientific review pending.'))
    except BaseException as e:
        write(root/'completion.json',dict(status='failed',finished_utc=utc(),reason=repr(e)));raise
    finally:
        write(root/'compute.json',dict(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            throughput_kind='inference; see per-condition summary',training_tokens=0))
        identity['end_utc']=utc();write(root/'identity.json',identity);fingerprint(root)


if __name__=='__main__':main()
