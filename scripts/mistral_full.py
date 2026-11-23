"""Full Mistral training, extending the pilot copy checkpoint without repeats."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json,file_sha256
from pathpred.end_to_end import DualBudget
from pathpred.planner_training import copy_batch,end_to_end_batch
from pathpred.planner_decode import decode_batch
from pathpred.budget_calibration import calibrate_threshold
from scripts import mistral_planner as mp,planner_experiment as pe
from scripts.neuron_guessing import data
from scripts.oracle_quality import aggregate,bootstrap_quality

ROOT=Path('artifacts/mistral-full-v1')
PILOT=Path('artifacts/mistral-pilot-v1')


def full_data(c):
    pilot_cfg=json.loads(Path('configs/mistral-planner-pilot.json').read_text())
    ids,rows,reserved,order=mp.training_order(pilot_cfg)
    extra,metadata=data(c['extra_train_corpus']);heldout,heldout_meta=data(c['heldout_corpus'])
    forbidden={r['document_sha256'] for r in rows+heldout_meta}
    assert not forbidden & {r['document_sha256'] for r in metadata}
    offset=len(rows)
    extension=[dict(r,sequence_id=r['sequence_id']+len(ids)) for r in metadata]
    indices=[i for i,r in enumerate(extension) if r['split']=='train' and r['length']>=512]
    indices.sort(key=lambda i:hashlib.sha256((str(c['seed'])+':'+extension[i]['document_sha256']+':'+str(extension[i]['document_token_offset'])).encode()).hexdigest())
    order=order+[i+offset for i in indices];rows=rows+extension
    ids=torch.cat((ids,extra['input_ids']))
    needed=sum(math.ceil(c[k+'_tokens']/512) for k in ('copy','scheduled','e2e'))+10
    assert len(order)>=needed,(len(order),needed)
    atomic_json(ROOT/'data-audit.json',{'available_full_windows':len(order),'required_windows':needed,
        'pilot_order_preserved':True,'reserved_threshold_windows':reserved,'heldout_hashes_excluded':True,
        'same_quality_data_both_budgets':True,'order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest()})
    return ids,rows,reserved,order


def fingerprint(c,config):
    files=list(Path('pathpred').glob('*.py'))+[config,Path(__file__),Path('scripts/mistral_planner.py'),
        Path('scripts/neuron_guessing.py'),Path('scripts/planner_experiment.py'),Path('scripts/oracle_quality.py'),
        Path(c['extra_train_corpus'])/'manifest.json',Path(c['train_corpus'])/'manifest.json',
        Path(c['heldout_corpus'])/'manifest.json',PILOT/'thresholds.safetensors']
    return {str(p):file_sha256(p) for p in sorted(set(files))}


def save_state(root,p,opt,offsets,dual,seen,cursor,elapsed,fp,selection):
    value={'model':p.state_dict(),'optimizer':opt.state_dict(),'offsets':offsets,
        'dual':None if dual is None else asdict(dual),'tokens':seen,'position':cursor,'elapsed':elapsed,
        'fingerprint':fp,'selection':selection,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
    tmp=root/'last.tmp';torch.save(value,tmp);tmp.replace(root/'last.pt')


def progress(token,total):
    if pe.PAUSE or (ROOT/'PAUSE').exists():raise SystemExit(75)
    print(json.dumps({'stage':'validation_decode','position':token,'total':total}),flush=True)


def update_selection(selection, seen, state, c):
    """Validation-only stopping; keep the best checkpoint even after hardening."""
    kl=state['overall']['teacher_kl_nats'];prior_best=selection['best_kl']
    meaningful=prior_best is None or prior_best-kl>=c['plateau_minimum_kl_improvement']
    selection['stale_checks']=0 if meaningful else selection['stale_checks']+1
    if prior_best is None or kl<prior_best:selection.update(best_kl=kl,best_tokens=seen)
    selection['curve'].append({'tokens':seen,'validation_kl':kl,
        'validation_ppl_increase':state['overall']['relative_ppl_increase'],
        'validation_kept':state['overall']['routed_kept']})
    if (not selection['hard_tail'] and seen>=c['plateau_minimum_tokens'] and seen<c['e2e_tokens']
            and selection['stale_checks']>=c['plateau_patience']):
        selection['hard_tail']=True
        selection['stop_at']=min(c['e2e_tokens'],seen+c['hardening_tail_tokens'])
    selection.pop('pending_validation',None)
    return selection


@torch.no_grad()
def score_documents(model,ctrl,p,q,offsets,corpus,rows,keep,c,prefix=2):
    result=[]
    for metadata,ids,lengths in mp.batches(corpus,rows,c['eval_batch_size']):
        z=decode_batch(model,ctrl,p,q,ids,lengths,offsets=offsets,keep=keep,progress=progress,execution_prefix=prefix)
        for i,r in enumerate(metadata):
            a=z['sums'][i].tolist();v=dict(zip(('teacher_nll','student_nll','kl','top1_agreement','tokens'),a))
            v.update(document=r['document_sha256'],domain=r['domain'],routed_kept=float(z['kept_sum'][i]/a[-1]));result.append(v)
    summary=aggregate(result);summary.update(bootstrap_quality(result,2000,c['seed']))
    summary['routed_kept']=sum(v['routed_kept']*v['tokens'] for v in result)/summary['tokens']
    return summary,result


@torch.no_grad()
def validate(model,ctrl,p,q,offsets,keep,c,root,prefix=2):
    root.mkdir(parents=True,exist_ok=True);p.eval()
    if (root/'validation.json').exists():return json.loads((root/'validation.json').read_text())
    tensors,metadata=data(c['heldout_corpus']);corpus=tensors['input_ids'];rows=[r for r in metadata if r['split']=='validation']
    trials_path=root/'calibration-trials.json';trials=json.loads(trials_path.read_text()) if trials_path.exists() else {}
    def measure(shift):
        key=repr(float(shift))
        if key in trials:return trials[key]
        count=kept=0.
        for _,ids,lengths in mp.batches(corpus,rows,c['eval_batch_size']):
            z=decode_batch(model,ctrl,p,q,ids,lengths,offsets=offsets+shift,keep=keep,quality=False,progress=progress,execution_prefix=prefix)
            count+=float(z['sums'][:,4].sum());kept+=float(z['kept_sum'].sum())
        trials[key]=kept/count;atomic_json(trials_path,trials)
        print(json.dumps({'stage':'validation_calibration','shift':shift,'kept':trials[key],'target':keep}),flush=True)
        return trials[key]
    if (root/'threshold.json').exists():cal=json.loads((root/'threshold.json').read_text())
    else:
        cal=calibrate_threshold(measure,keep,tolerance=c['calibration_tolerance'],max_evaluations=c['calibration_max_evaluations'])
        cal['offsets']=(offsets+cal['shift']).cpu().tolist();atomic_json(root/'threshold.json',cal)
    calibrated=torch.tensor(cal['offsets'],device='cuda')
    summary,documents=score_documents(model,ctrl,p,q,calibrated,corpus,rows,keep,c,prefix)
    value={'overall':summary,'offsets':cal['offsets'],'calibration':cal,'execution_prefix':prefix}
    atomic_json(root/'validation-documents.json',documents);atomic_json(root/'validation.json',value)
    return value


def train(c,config,stage,budget):
    quality=stage=='e2e';root=ROOT/(f'budget-{budget}' if quality else stage);root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():return
    fp=fingerprint(c,config);model,ctrl=mp.load_model(c);p=mp.make_planner(c)
    q=load_file(str(PILOT/'thresholds.safetensors'),device='cuda')['q75'];corpus,rows,reserved,order=full_data(c)
    lr=c['quality_learning_rate'] if quality else c['copy_learning_rate'];opt=torch.optim.AdamW(p.parameters(),lr=lr,weight_decay=c['weight_decay'])
    offsets=torch.zeros(p.steps,device='cuda');dual=DualBudget(budget/100,c['alpha_initial'],c['dual_learning_rate'],c['maximum_alpha']) if quality else None
    selection={'best_kl':None,'best_tokens':None,'stale_checks':0,'stop_at':c[stage+'_tokens'],'hard_tail':False,'curve':[]}
    seen=cursor=last_save=0;elapsed=0.
    if (root/'last.pt').exists():
        saved=torch.load(root/'last.pt',map_location='cuda',weights_only=True);assert saved['fingerprint']==fp
        p.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);offsets=saved['offsets'];seen=saved['tokens'];cursor=saved['position'];elapsed=saved['elapsed'];last_save=seen
        dual=DualBudget(**saved['dual']) if quality else None;selection=saved['selection']
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());del saved
    elif stage=='copy':
        # Preserve optimizer and RNG as well as the 1M already paid-for examples.
        saved=torch.load(PILOT/'copy/last.pt',map_location='cuda',weights_only=True)
        meta=json.loads((PILOT/'copy/checkpoint-01000000.json').read_text())
        assert file_sha256(PILOT/'copy'/meta['file']['file'])==meta['file']['sha256']
        assert saved['tokens']==1000000
        p.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);seen=saved['tokens'];cursor=saved['position'];elapsed=saved['elapsed']
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());del saved
        atomic_json(root/'warm-start.json',{'source':str(PILOT/'copy'),'reused_tokens':seen,'next_cursor':cursor,
            'source_checkpoint_sha256':meta['file']['sha256'],'learning_rate_schedule':'Resume at cumulative tokens on extended 10M cosine schedule'})
    else:
        previous='scheduled' if quality else 'copy';done=json.loads((ROOT/previous/'completion.json').read_text())
        pe.load_stage_weights(p,ROOT/previous,done['tokens']);cursor=done['next_cursor']
        if quality:offsets=pe.training_offsets(model,ctrl,p,q,corpus,rows,reserved[:4],budget/100)
    total=c[stage+'_tokens'];started=time.monotonic();measured=0.;new_tokens=0;torch.cuda.reset_peak_memory_stats()
    def finish_validation():
        pe.save_weights(root,p,seen,fp)
        state=validate(model,ctrl,p,q,offsets,budget/100,c,root/f'check-{seen:08d}')
        update_selection(selection,seen,state,c)
        atomic_json(root/'selection.json',selection)
        save_state(root,p,opt,offsets,dual,seen,cursor,elapsed+time.monotonic()-started,fp,selection)
    # Resume an interrupted calibration before consuming another training window.
    if quality and selection.get('pending_validation') is not None:
        assert selection['pending_validation']==seen
        finish_validation()
    while seen<selection['stop_at']:
        milestone=min([m for m in c['quality_checkpoint_tokens'] if m>seen]+[selection['stop_at']]) if quality else total
        length=min(512,milestone-seen,selection['stop_at']-seen);assert length>0
        ids=pe.ids_for(corpus,rows,order[cursor],length);factor=min(1.,(seen+length)/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/total)))
        for group in opt.param_groups:group['lr']=lr*factor
        torch.cuda.synchronize();tick=time.monotonic()
        if quality:
            fraction=seen/total;mode='hard' if selection['hard_tail'] or fraction>=1-c['hard_fraction'] else 'gumbel'
            temperature=c['temperatures'][0]*(c['temperatures'][1]/c['temperatures'][0])**min(1.,fraction/(1-c['hard_fraction']))
            metrics=end_to_end_batch(model,ctrl,p,opt,q,ids,offsets,dual,mode=mode,temperature=temperature,beta=c['beta'],epsilon=c['epsilon'],chunk_size=c['teacher_head_chunk_size'])
        else:metrics=copy_batch(model,ctrl,p,opt,q,ids,keep=.5 if cursor%2==0 else .4,scheduled=stage=='scheduled',teacher_probability=max(0.,1-seen/total),positive_weight=c['positive_weight'])
        torch.cuda.synchronize();measured+=time.monotonic()-tick;seen+=length;new_tokens+=length;cursor+=1
        record={'stage':stage,'budget':budget if quality else None,'tokens':seen,'target_tokens':selection['stop_at'],
            'next_cursor':cursor,'tokens_per_second':new_tokens/measured,'seconds':elapsed+time.monotonic()-started,'metrics':metrics}
        if cursor%20==0 or seen==milestone:
            atomic_json(root/'progress.json',record);print(json.dumps(record),flush=True)
            with (root/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        pause=pe.PAUSE or (ROOT/'PAUSE').exists();checkpoint=seen-last_save>=c['checkpoint_every_tokens'] or seen==milestone or pause
        if quality and seen==milestone:selection['pending_validation']=seen
        if checkpoint:
            save_state(root,p,opt,offsets,dual,seen,cursor,elapsed+time.monotonic()-started,fp,selection);last_save=seen
        if pause:raise SystemExit(75)
        if quality and seen==milestone:
            finish_validation()
    pe.save_weights(root,p,seen,fp)
    atomic_json(root/'completion.json',{'status':'complete','tokens':seen,'next_cursor':cursor,'seconds':elapsed+time.monotonic()-started,
        'offsets':offsets.cpu().tolist(),'selection':selection,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'backbone_frozen':all(not v.requires_grad and v.grad is None for v in model.parameters()),'fingerprint':fp})


def selected(c,budget):
    root=ROOT/f'budget-{budget}';done=json.loads((root/'completion.json').read_text());tokens=done['selection']['best_tokens']
    model,ctrl=mp.load_model(c);p=mp.make_planner(c);pe.load_stage_weights(p,root,tokens);p.eval()
    q=load_file(str(PILOT/'thresholds.safetensors'),device='cuda')['q75']
    state=json.loads((root/f'check-{tokens:08d}'/'validation.json').read_text())
    return model,ctrl,p,q,torch.tensor(state['offsets'],device='cuda'),tokens


@torch.no_grad()
def evaluate(c,budget):
    root=ROOT/f'budget-{budget}'
    if (root/'evaluation.json').exists():return
    model,ctrl,p,q,offsets,tokens=selected(c,budget)
    tensors,records=data(c['heldout_corpus']);test=[r for r in records if r['split']=='test']
    overall,rows=score_documents(model,ctrl,p,q,offsets,tensors['input_ids'],test,budget/100,c)
    atomic_json(root/'evaluation-documents.json',rows)
    atomic_json(root/'evaluation.json',{'overall':overall,'selected_tokens':tokens,
        'domains':{d:aggregate([r for r in rows if r['domain']==d]) for d in ('web','code','conversation')},
        'selection':'Best validation KL at validation-calibrated budget; test used only after selection'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','copy','scheduled','e2e','evaluate']);parser.add_argument('--budget',type=int,default=50)
    parser.add_argument('--config',type=Path,default=Path('configs/mistral-planner-full.json'));args=parser.parse_args()
    c=json.loads(args.config.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGTERM,pe.request_pause);signal.signal(signal.SIGUSR1,pe.request_pause)
    if args.stage=='audit':full_data(c)
    elif args.stage=='evaluate':evaluate(c,args.budget)
    else:train(c,args.config,args.stage,args.budget)


if __name__=='__main__':main()
