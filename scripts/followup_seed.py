"""Exact original Qwen planner recipe with a new initialization, fixed data order."""
import argparse,gc,json,sys,time,shutil
from pathlib import Path
import torch
from safetensors.torch import load_file
from scripts import planner_experiment as pe
from scripts.followup_common import read,write,utc,start_record,fingerprint,ci
from scripts.oracle_quality import aggregate


def evaluate_imitation(c,config,variant):
 root=pe.ROOT/variant/'copy';model,ctrl=pe.backbone();q=load_file(str(pe.CAL),device='cuda')['q75'];p=pe.planner(variant,c)
 pe.load_stage_weights(p,root,c['copy_tokens']);p.eval()
 corpus,_,test=pe.selections();_,records=pe.data('data/stage3-corpus-v1')
 validation=pe.select_documents(records,'validation',c['validation_documents_per_domain'],{'seed':20261111,'minimum_sequence_length':128})
 assert not {r['document_sha256'] for r in validation}&{r['document_sha256'] for r in test}
 for budget in c['budgets']:
  out=root/f'eval-budget-{budget}';out.mkdir(parents=True,exist_ok=True)
  offsets=pe.calibrate(model,ctrl,p,q,torch.zeros(p.steps,device='cuda'),corpus,validation,budget/100,out/'threshold.json',c)
  pe.evaluate(model,ctrl,p,q,offsets,corpus,test,budget/100,out/'evaluation.json',c)
 del p,model;gc.collect();torch.cuda.empty_cache()


def prepare_first_seed(c,record):
 from pathpred.collection import file_sha256
 base=pe.ROOT/'seed1-reference'
 for variant in c['variants']:
  meta=read(record/'historical-seed1'/variant/'copy/checkpoint-10000000.json')
  candidates=list(Path('/workspace').glob('*/artifacts/planner-v1/'+variant+'/copy/'+meta['file']['file']))
  source=next((p for p in candidates if file_sha256(p)==meta['file']['sha256']),None)
  if source is None:raise FileNotFoundError('Original imitation checkpoint unavailable: '+variant)
  dest=base/variant/'copy';dest.mkdir(parents=True,exist_ok=True)
  shutil.copy2(record/'historical-seed1'/variant/'copy/checkpoint-10000000.json',dest/'checkpoint-10000000.json')
  if not (dest/meta['file']['file']).exists():(dest/meta['file']['file']).symlink_to(source.resolve())
 return base


def summarize(c):
 summary={};documents={}
 for variant in c['variants']:
  for budget in c['budgets']:
   for stage,relative in [('imitation',f'copy/eval-budget-{budget}'),('e2e',f'budget-{budget}')]:
    key=f'{variant}-{stage}-{budget}';root=pe.ROOT/variant/relative;v=read(root/'evaluation.json');rows=read(root/'evaluation-documents.json')
    def metric(rs,k):
     if k=='routed_kept':return sum(r['routed_kept']*r['tokens'] for r in rs)/sum(r['tokens'] for r in rs)
     return aggregate(rs)[k]
    result=v['overall']
    for k in ['dense_perplexity','student_perplexity','relative_ppl_increase','teacher_kl_nats','top1_agreement','routed_kept']:
     result[k+'_ci95']=ci(rows,lambda rs:metric(rs,k))
    result.update(training_seed=c['seed'],data_order_seed=c['data_seed'],single_seed=True,domains=v['domains'],bootstrap='paired document percentile bootstrap; evaluation noise, not seed variation')
    calibration=read(root/'threshold.json')
    result.update(validation_retention=calibration['validation_retention'],calibration_tolerance=calibration['tolerance'],validation_retention_matched=calibration['absolute_error']<=calibration['tolerance'])
    summary[key]=result;documents[key]=rows
 paired={}
 for budget in c['budgets']:
  pairs=[(f'autoregressive-e2e-{budget}',f'one_shot_history-e2e-{budget}')]+[(f'{v}-e2e-{budget}',f'{v}-imitation-{budget}') for v in c['variants']]
  for aa,bb in pairs:
   a={x['document']:x for x in documents[aa]};b={x['document']:x for x in documents[bb]};assert a.keys()==b.keys()
   rows=[(a[k],b[k]) for k in sorted(a)]
   diff=lambda rs:aggregate([x[0] for x in rs])['relative_ppl_increase']-aggregate([x[1] for x in rs])['relative_ppl_increase']
   paired[aa+'-minus-'+bb]=dict(relative_ppl_gap=diff(rows),ci95=ci(rows,diff),interpretation='Negative favors the first arm; each seed remains separately labeled.')
 write(pe.ROOT/'summary.json',dict(conditions=summary,paired=paired))


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=read(a.config);root=Path(c['collection']);pe.ROOT=Path(c['output']);pe.ROOT.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 identity=start_record(root,sys.argv);fingerprint(root);start=time.time();runroot=pe.ROOT
 try:
  if c.get('resume_artifact_source'):
   from scripts.followup_resume import inherit_qwen
   inherit_qwen(c,root,pe.ROOT)
  reference_root=prepare_first_seed(c,root)
  for quality in [False,True]:
   pe.ordered_data(c,quality=quality);audit=read(pe.ROOT/('quality-data-audit.json' if quality else 'copy-data-audit.json'))
   reference=read(root/('original-quality-data-audit.json' if quality else 'original-copy-data-audit.json'))
   assert audit['order_sha256']==reference['order_sha256'],'Training example/order mismatch'
   assert audit['reserved_calibration_windows']==reference['reserved_calibration_windows']
  stages=[]
  for variant in c['variants']:
   for stage,budget in [('copy',50),('scheduled',50),('e2e',50),('e2e',40)]:
    where=pe.ROOT/variant/(f'budget-{budget}' if stage=='e2e' else stage)
    if variant in c.get('resume_completed_variants',[]):
     assert read(where/'completion.json')['status']=='complete'
    else:pe.train(c,Path(c.get('training_fingerprint_config',a.config)),variant,stage,budget)
    stages.append(dict(variant=variant,stage=stage,budget=budget if stage=='e2e' else None,path=str(where),training=read(where/'training.json'),completion=read(where/'completion.json'),optimizer='reset between stages; saved and restored only for same-stage resume',rng='new fixed stage seed; saved and restored on resume'))
    write(pe.ROOT/'training-record.json',dict(stages=stages,seed=c['seed'],data_seed=c['data_seed'],history_files=[str(x.relative_to(pe.ROOT)) for x in pe.ROOT.rglob('history.jsonl')]))
   evaluate_imitation(c,a.config,variant)
  summarize(c)
  primary=read(pe.ROOT/'summary.json')
  for variant in c['variants']:
   pe.ROOT=reference_root;old=dict(c,seed=c['data_seed']);evaluate_imitation(old,a.config,variant)
  pe.ROOT=runroot
  first_seed_documents={}
  for variant in c['variants']:
   for budget in c['budgets']:
    q=read(reference_root/variant/f'copy/eval-budget-{budget}/evaluation.json')['overall'];q.update(training_seed=c['data_seed'],source='new evaluation of archived first-seed imitation checkpoint')
    cal=read(reference_root/variant/f'copy/eval-budget-{budget}/threshold.json');q.update(validation_retention_matched=cal['absolute_error']<=cal['tolerance'],validation_retention=cal['validation_retention'],calibration_tolerance=cal['tolerance'])
    primary['conditions'][f'seed1-{variant}-imitation-{budget}']=q
    old=read(root/f'historical-seed1/{variant}/budget-{budget}/evaluation.json')['overall'];old.update(training_seed=c['data_seed'],source='historical first-seed end-to-end evaluation, same declared protocol')
    primary['conditions'][f'seed1-{variant}-e2e-{budget}']=old
    first=read(reference_root/variant/f'copy/eval-budget-{budget}/evaluation-documents.json');second=read(root/f'historical-seed1/{variant}/budget-{budget}/evaluation-documents.json');aa={x['document']:x for x in first};bb={x['document']:x for x in second};assert aa.keys()==bb.keys()
    for stage,rows in [('imitation',first),('e2e',second)]:
     key=f'seed1-{variant}-{stage}-{budget}';first_seed_documents[key]=rows
     for metric in ['dense_perplexity','student_perplexity','relative_ppl_increase','teacher_kl_nats','top1_agreement','routed_kept']:
      fn=(lambda rs:sum(r['routed_kept']*r['tokens'] for r in rs)/sum(r['tokens'] for r in rs)) if metric=='routed_kept' else (lambda rs,k=metric:aggregate(rs)[k])
      primary['conditions'][key][metric+'_ci95']=ci(rows,fn)
     primary['conditions'][key].update(single_seed=True,bootstrap='paired document percentile bootstrap; evaluation noise, not seed variation')
    pairs=[(bb[k],aa[k]) for k in sorted(aa)];diff=lambda rs:aggregate([x[0] for x in rs])['relative_ppl_increase']-aggregate([x[1] for x in rs])['relative_ppl_increase']
    primary['paired'][f'seed1-{variant}-e2e-minus-imitation-{budget}']=dict(relative_ppl_gap=diff(pairs),ci95=ci(pairs,diff))
  replication={}
  for budget in c['budgets']:
   aa={r['document']:r for r in first_seed_documents[f'seed1-autoregressive-e2e-{budget}']};bb={r['document']:r for r in first_seed_documents[f'seed1-one_shot_history-e2e-{budget}']};assert aa.keys()==bb.keys()
   pairs=[(aa[k],bb[k]) for k in sorted(aa)];diff=lambda rs:aggregate([x[0] for x in rs])['relative_ppl_increase']-aggregate([x[1] for x in rs])['relative_ppl_increase']
   primary['paired'][f'seed1-autoregressive-minus-one-shot-{budget}']=dict(relative_ppl_gap=diff(pairs),ci95=ci(pairs,diff))
   pairs_to_check=[('planner-vs-one-shot',f'seed1-autoregressive-minus-one-shot-{budget}',f'autoregressive-e2e-{budget}-minus-one_shot_history-e2e-{budget}')]+[(v+'-e2e-vs-imitation',f'seed1-{v}-e2e-minus-imitation-{budget}',f'{v}-e2e-{budget}-minus-{v}-imitation-{budget}') for v in c['variants']]
   for label,old,new in pairs_to_check:
    x=primary['paired'][old]['relative_ppl_gap'];y=primary['paired'][new]['relative_ppl_gap'];replication[f'{label}-{budget}']=dict(first_seed_gap=x,second_seed_gap=y,favors_first_arm_in_both_seeds=x<0 and y<0,interpretation='Descriptive two-seed direction check; no population-level seed uncertainty claim.')
  primary['replication']=replication
  primary['calibration_matching_complete']=all(v.get('validation_retention_matched',True) for v in primary['conditions'].values())
  if not primary['calibration_matching_complete']:
   for v in primary['replication'].values():v['interpretation']='Unmatched validation retention in at least one arm: descriptive only; matched-retention replication conclusion withheld.'
  write(pe.ROOT/'summary.json',primary)
  write(pe.ROOT/'calibration-index.json',dict(files=[str(x.relative_to(pe.ROOT)) for x in pe.ROOT.rglob('*threshold*.json')],selection='validation retention only; every trial preserved'))
  write(root/'completion.json',dict(status='complete',reason='Both architectures and both budgets completed with fixed original data order.',finished_utc=utc()))
 except BaseException as e:
  write(root/'completion.json',dict(status='failed',reason=repr(e),finished_utc=utc()));raise
 finally:
  pe.ROOT=runroot
  write(root/'compute.json',dict(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),total_wall_seconds=time.time()-start,throughput_kind='training; per-stage tokens and seconds in raw/training-record.json; no inference speed claim'))
  identity['end_utc']=utc();write(root/'identity.json',identity);fingerprint(root)
if __name__=='__main__':main()
