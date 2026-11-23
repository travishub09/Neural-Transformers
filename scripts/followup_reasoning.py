"""Full MMLU/GSM8K with sparse answer prediction and fixed few-shot prompts."""
import argparse,gzip,hashlib,json,re,sys,time
from decimal import Decimal,InvalidOperation
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from scripts.followup_common import backbone,planner_policy,start_record,fingerprint,read,write,utc,ci
from scripts.mistral_comparison_core import Policy
from scripts.followup_speculative import quality_summary


def number(text,strict=True):
 pattern=r'####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)' if strict else r'[-+]?[0-9][0-9,]*(?:\.[0-9]+)?'
 found=re.findall(pattern,text)
 if not found:return None
 try:return str(Decimal(found[-1].replace(',','')).normalize())
 except InvalidOperation:return None


@torch.inference_mode()
def first_prediction(pol,rows):
 # Process final prompt input sparsely so single-letter MMLU genuinely tests
 # the policy; every earlier prefix token remains dense.
 device=next(pol.model.parameters()).device;prompts=[r['prompt_ids'] for r in rows];n=max(len(x)-1 for x in prompts)
 ids=torch.zeros(len(rows),n,dtype=torch.long,device=device);valid=torch.zeros_like(ids,dtype=torch.bool)
 for i,x in enumerate(prompts):ids[i,-(len(x)-1):]=torch.tensor(x[:-1],device=device);valid[i,-(len(x)-1):]=True
 pol.reset();pol.prefill(ids,valid);valid=torch.cat([valid,torch.ones(len(rows),1,device=device,dtype=torch.bool)],1)
 z=pol.step(torch.tensor([[x[-1]] for x in prompts],device=device),valid)
 masks=pol.masks();ret=torch.ones(len(rows),device=device) if masks is None else masks.float().mean((1,2))
 return z,valid,ret


@torch.inference_mode()
def generate(pol,rows,tok,maximum):
 z,valid,ret=first_prediction(pol,rows);device=z.device;active=torch.ones(len(rows),dtype=torch.bool,device=device)
 tokens=[[] for _ in rows];kept=torch.zeros(len(rows),device=device);counts=torch.zeros_like(kept);reasons=['length']*len(rows)
 for t in range(maximum):
  kept+=ret*active;counts+=active;chosen=z.argmax(-1);values=chosen.tolist();live=active.tolist()
  for i,(v,on) in enumerate(zip(values,live)):
   if not on:continue
   tokens[i].append(v)
   if v==tok.eos_token_id:active[i]=False;reasons[i]='eos'
   elif '\nQuestion:' in tok.decode(tokens[i][-10:],skip_special_tokens=True):active[i]=False;reasons[i]='next_question'
  if not active.any() or t+1==maximum:break
  valid=torch.cat([valid,torch.ones(len(rows),1,device=device,dtype=torch.bool)],1);z=pol.step(chosen[:,None],valid)
  masks=pol.masks();ret=torch.ones(len(rows),device=device) if masks is None else masks.float().mean((1,2))
 result=[]
 for i,row in enumerate(rows):
  text=tok.decode(tokens[i],skip_special_tokens=True);gold=number(row['gold'],False)
  result.append(dict(id=row['id'],document=row['document_sha256'],domain='gsm8k',tokens=tokens[i],text=text,gold=gold,
    strict_prediction=number(text),flexible_prediction=number(text,False),accuracy=int(number(text)==gold),
    accuracy_flexible=int(number(text,False)==gold),stop_reason=reasons[i],scored_executions=int(counts[i]),kept_sum=float(kept[i]),routed_kept=float(kept[i]/counts[i])))
 return result


@torch.inference_mode()
def reference_quality(policies,rows,tok):
 # Fixed test subset; reference answers never select thresholds or checkpoints.
 answers=[]
 for r in rows:
  full=tok.encode(r['prompt']+' '+r['source']['answer'],add_special_tokens=False)
  assert full[:len(r['prompt_ids'])]==r['prompt_ids'],'Reference-answer token boundary changed'
  answers.append(full[len(r['prompt_ids']):])
 states={k:first_prediction(p,rows) for k,p in policies.items()};device=states['dense'][0].device
 sums={k:torch.zeros(len(rows),5,device=device,dtype=torch.float64) for k in policies};kept={k:torch.zeros(len(rows),device=device) for k in policies};counts=torch.zeros(len(rows),device=device)
 for t in range(max(map(len,answers))):
  live=torch.tensor([t<len(x) for x in answers],device=device);target=torch.tensor([x[t] if t<len(x) else 0 for x in answers],device=device)
  logs={k:z[0].float().log_softmax(-1) for k,z in states.items()};teacher=logs['dense'];counts+=live
  for k,p in policies.items():
   lp=logs[k];values=torch.stack([-teacher.gather(-1,target[:,None])[:,0],-lp.gather(-1,target[:,None])[:,0],(teacher.exp()*(teacher-lp)).sum(-1),(teacher.argmax(-1)==lp.argmax(-1)).float(),torch.ones(len(rows),device=device)],1)
   sums[k]+=values.double()*live[:,None];kept[k]+=states[k][2]*live
   if t+1<max(map(len,answers)):
    valid=torch.cat([states[k][1],torch.ones(len(rows),1,device=device,dtype=torch.bool)],1);z=p.step(target[:,None],valid);m=p.masks();states[k]=(z,valid,torch.ones(len(rows),device=device) if m is None else m.float().mean((1,2)))
 return {k:[dict(zip(['teacher_nll','student_nll','kl','top1_agreement','tokens'],sums[k][i].tolist()),document=r['document_sha256'],domain='gsm8k',kept_sum=float(kept[k][i]),executions=float(counts[i])) for i,r in enumerate(rows)] for k in policies}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=read(a.config);root=Path(c['collection']);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;identity=start_record(root,sys.argv);fingerprint(root);start=time.time()
 try:
  manifest=read(root/'source-data-manifest.json');datasets={}
  for task in ['gsm8k','mmlu']:
   data=gzip.decompress((root/'inputs'/f'{task}.jsonl.gz').read_bytes());assert hashlib.sha256(data).hexdigest()==manifest['sources'][task]['file_sha256']
   datasets[task]=[json.loads(x) for x in data.splitlines()];assert len(datasets[task])==c['examples'][task]
  tok=AutoTokenizer.from_pretrained('checkpoints/mistral-7b-v0.1',local_files_only=True);model,ctrl=backbone();policies={'dense':Policy(model,ctrl,'dense')}
  for budget in [30,20]:pol,prov=planner_policy(model,ctrl,budget);policies[f'planner-{budget}']=pol;write(root/f'checkpoint-{budget}.json',prov)
  records={task:{k:[] for k in policies} for task in datasets}
  if c.get('resume_collection'):
   from scripts.followup_resume import inherit_reasoning
   inherit_reasoning(c,root,out)
   for task in records:
    expected={r['id']:r['document_sha256'] for r in datasets[task]}
    for k in policies:
     p=out/f'{task}-{k}-examples.json'
     if p.exists():
      records[task][k]=read(p)
      assert len({r['id'] for r in records[task][k]})==len(records[task][k])
      assert all(expected[r['id']]==r['document'] for r in records[task][k])
  # MMLU has full-candidate vocab distributions and one scored answer per row.
  ordered=sorted(datasets['mmlu'],key=lambda x:(len(x['prompt_ids']),x['id']))
  for begin in range(0,len(ordered),c['batch_size']):
   batch=ordered[begin:begin+c['batch_size']]
   existing={k:{r['id'] for r in records['mmlu'][k]} for k in policies}
   needed=[k for k in policies if any(r['id'] not in existing[k] for r in batch)]
   if not needed:continue
   zs={k:first_prediction(p,batch) for k,p in policies.items() if k in needed or k=='dense'};teacher=zs['dense'][0].float().log_softmax(-1)
   for k,(z,valid,ret) in zs.items():
    lp=z.float().log_softmax(-1)
    for i,row in enumerate(batch):
     if row['id'] in existing[k]:continue
     ids=row['choice_token_ids'];scores=lp[i,ids].tolist();gold=ids[row['gold']]
     records['mmlu'][k].append(dict(id=row['id'],document=row['document_sha256'],domain=row['domain'],scores=scores,gold=row['gold'],prediction=int(np.argmax(scores)),accuracy=int(np.argmax(scores)==row['gold']),teacher_nll=float(-teacher[i,gold]),student_nll=float(-lp[i,gold]),kl=float((teacher[i].exp()*(teacher[i]-lp[i])).sum()),top1_agreement=float(teacher[i].argmax()==lp[i].argmax()),tokens=1,executions=1,kept_sum=float(ret[i]),routed_kept=float(ret[i])))
   if begin%128==0 or begin+len(batch)==len(ordered):
    for k in policies:write(out/f'mmlu-{k}-examples.json',records['mmlu'][k])
    print(json.dumps(dict(task='mmlu',examples=begin+len(batch))),flush=True)
  ordered=sorted(datasets['gsm8k'],key=lambda x:(len(x['prompt_ids']),x['id']))
  for begin in range(0,len(ordered),c['batch_size']):
   batch=ordered[begin:begin+c['batch_size']]
   for k,p in policies.items():
    existing={r['id'] for r in records['gsm8k'][k]}
    if all(r['id'] in existing for r in batch):continue
    records['gsm8k'][k].extend(r for r in generate(p,batch,tok,c['max_new_tokens']) if r['id'] not in existing)
    write(out/f'gsm8k-{k}-examples.json',records['gsm8k'][k])
   print(json.dumps(dict(task='gsm8k',examples=begin+len(batch))),flush=True)
  subset=sorted(datasets['gsm8k'],key=lambda r:r['id'])[:64];quality={k:[] for k in policies}
  for begin in range(0,len(subset),c['batch_size']):
   value=reference_quality(policies,subset[begin:begin+c['batch_size']],tok)
   for k in policies:quality[k].extend(value[k]);write(out/f'gsm8k-reference-{k}-documents.json',quality[k])
  summary={}
  for task,conditions in records.items():
   dense={r['id']:r for r in conditions['dense']}
   for name,rows in conditions.items():
    values=[r['accuracy'] for r in rows];differences=[r['accuracy']-dense[r['id']]['accuracy'] for r in rows]
    q=quality_summary(rows if task=='mmlu' else quality[name])
    for metric in ['dense_perplexity','student_perplexity','relative_ppl_increase','teacher_kl_nats','top1_agreement','routed_kept']:
     q[metric+'_ci95']=ci(rows if task=='mmlu' else quality[name],lambda rs:quality_summary(rs)[metric])
    domains={d:float(np.mean([r['accuracy'] for r in rows if r['domain']==d])) for d in sorted({r['domain'] for r in rows})}
    summary[f'{task}-{name}']=dict(examples=len(rows),accuracy=float(np.mean(values)),accuracy_ci95=ci(values,lambda xs:float(np.mean(xs))),paired_accuracy_gap_vs_dense=float(np.mean(differences)),paired_accuracy_gap_ci95=ci(differences,lambda xs:float(np.mean(xs))),routed_kept=sum(r['kept_sum'] for r in rows)/sum(r.get('executions',r.get('scored_executions')) for r in rows),domains=domains,macro_subject_accuracy=float(np.mean(list(domains.values()))),quality=q,quality_protocol='all MMLU gold letter predictions' if task=='mmlu' else 'first 64 official test reference answers; full answer teacher forcing, separate from generated accuracy',single_seed=True,bootstrap='paired example percentile bootstrap; evaluation noise, not seed variation')
    if task=='gsm8k':
     summary[f'{task}-{name}'].update(flexible_accuracy=float(np.mean([r['accuracy_flexible'] for r in rows])),stop_reason_counts={reason:sum(r['stop_reason']==reason for r in rows) for reason in sorted({r['stop_reason'] for r in rows})})
  write(out/'summary.json',dict(conditions=summary))
  write(root/'completion.json',dict(status='complete',reason='Full benchmark splits and fixed reference-quality audit completed.',finished_utc=utc()))
 except BaseException as e:
  write(root/'completion.json',dict(status='failed',reason=repr(e),finished_utc=utc()));raise
 finally:
  write(root/'compute.json',dict(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.time()-start,throughput_kind='downstream quality evaluation; no inference speed claim'))
  identity['end_utc']=utc();write(root/'identity.json',identity);fingerprint(root)
if __name__=='__main__':main()
