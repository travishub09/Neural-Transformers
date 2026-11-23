"""One Mistral planner lineage: 25M at 50%, then 10–15M at 40%."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json,file_sha256
from pathpred.end_to_end import DualBudget
from pathpred.planner_training import end_to_end_batch
from scripts import mistral_full as mf,mistral_planner as mp,planner_experiment as pe
from scripts.neuron_guessing import data
from scripts.oracle_quality import aggregate

ROOT=Path('artifacts/mistral-phased-v1')
CONFIG=Path('configs/mistral-planner-phased.json')


def full_data(c):
    old=json.loads(Path('configs/mistral-planner-full.json').read_text())
    ids,rows,reserved,order=mf.full_data(old)
    prior_order_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    extra,metadata=data(c['additional_train_corpus']);_,heldout=data(c['heldout_corpus'])
    assert not {r['document_sha256'] for r in metadata}&{r['document_sha256'] for r in rows+heldout}
    extension=[dict(r,sequence_id=r['sequence_id']+len(ids)) for r in metadata]
    indices=[i for i,r in enumerate(extension) if r['split']=='train' and r['length']>=512]
    indices.sort(key=lambda i:hashlib.sha256((str(c['seed'])+':'+extension[i]['document_sha256']+':'+str(extension[i]['document_token_offset'])).encode()).hexdigest())
    order=order+[i+len(rows) for i in indices];rows=rows+extension;ids=torch.cat((ids,extra['input_ids']))
    needed=sum(math.ceil(n/512) for n in [c['copy_tokens'],c['scheduled_tokens'],*c['phase_tokens'].values()])+100
    assert len(order)>=needed,(len(order),needed)
    atomic_json(ROOT/'data-audit.json',{'available_full_windows':len(order),'required_windows':needed,
        'prior_order_sha256':prior_order_hash,'order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        'old_order_preserved':True,'new_document_hashes_disjoint':True,'phase40_continues_data_cursor':True})
    return ids,rows,reserved,order


def fingerprint(c):
    result=mf.fingerprint(c,CONFIG)
    for path in [Path(__file__),Path(c['additional_train_corpus'])/'manifest.json']:
        result[str(path)]=file_sha256(path)
    return result


def migration_state(saved,c,budget):
    """Preserve weights, Adam moments, RNG and cursor; explicitly reset phase policy."""
    migrated=dict(saved)
    original_tokens=saved['tokens']
    if budget==50:
        assert 0<original_tokens<c['phase_tokens']['50']
        migrated['schedule_origin_tokens']=original_tokens
        migrated['selection']=dict(saved['selection'],stop_at=c['phase_tokens']['50'],hard_tail=False)
        migrated['selection']['best_kl']=None
        migrated['selection']['best_tokens']=None
        migrated['selection']['stale_checks']=0
    else:
        assert original_tokens==c['phase_tokens']['50']
        migrated.update(tokens=0,elapsed=0.,schedule_origin_tokens=0)
        migrated['selection']={'curve':[],'stop_at':c['phase_tokens']['40'],'hard_tail':False}
        migrated['dual']=asdict(DualBudget(.4,c['alpha_initial'],c['dual_learning_rate'],c['maximum_alpha']))
    return migrated


def schedule(c,budget,seen,origin):
    total=c['phase_tokens'][str(budget)];base=c['quality_learning_rate']
    if budget==50:
        # Preserve the old schedule's value at the transfer; stretch its remaining tail.
        old_total=20_000_000
        start_factor=.1+.45*(1+math.cos(math.pi*origin/old_total))
        f=max(0.,min(1.,(seen-origin)/(total-origin)))
        factor=.1+(start_factor-.1)*.5*(1+math.cos(math.pi*f))
        hard_start=total*(1-c['hard_fraction'])
        start_temp=c['temperatures'][0]*(c['temperatures'][1]/c['temperatures'][0])**min(1.,origin/(old_total*(1-c['hard_fraction'])))
        tf=max(0.,min(1.,(seen-origin)/max(1,hard_start-origin)))
        temperature=start_temp*(c['temperatures'][1]/start_temp)**tf
    else:
        factor=min(1.,(seen+512)/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/total)))
        hard_start=c['phase40_hard_start_tokens']
        temperature=c['temperatures'][0]*(c['temperatures'][1]/c['temperatures'][0])**min(1.,seen/hard_start)
    return base*factor,temperature,'hard' if seen>=hard_start else 'gumbel'


def update_selection(s,seen,state,c,budget):
    kl=state['overall']['teacher_kl_nats']
    s['curve'].append({'tokens':seen,'validation_kl':kl,
        'validation_ppl_increase':state['overall']['relative_ppl_increase'],
        'validation_kept':state['overall']['routed_kept']})
    if budget==40 and seen==c['minimum_phase40_tokens']:
        previous=[x['validation_kl'] for x in s['curve'] if x['tokens']<seen]
        improvement=min(previous)-kl if previous else float('inf')
        s['extension_decision']={'at_tokens':seen,'improvement_over_best_previous_kl':improvement,
            'extend_to_15M':improvement>=c['plateau_minimum_kl_improvement']}
        if improvement<c['plateau_minimum_kl_improvement']:s['stop_at']=seen
    s.pop('pending_validation',None)


def save(root,p,opt,offsets,dual,seen,cursor,elapsed,fp,selection,origin):
    value={'model':p.state_dict(),'optimizer':opt.state_dict(),'offsets':offsets,'dual':asdict(dual),
        'tokens':seen,'position':cursor,'elapsed':elapsed,'fingerprint':fp,'selection':selection,
        'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'schedule_origin_tokens':origin}
    tmp=root/'last.tmp';torch.save(value,tmp);tmp.replace(root/'last.pt')


def train(c,budget):
    root=ROOT/f'budget-{budget}';root.mkdir(parents=True,exist_ok=True)
    if (root/'completion.json').exists():return
    model,ctrl=mp.load_model(c);p=mp.make_planner(c);fp=fingerprint(c)
    q=load_file(str(mf.PILOT/'thresholds.safetensors'),device='cuda')['q75']
    corpus,rows,reserved,order=full_data(c)
    opt=torch.optim.AdamW(p.parameters(),lr=c['quality_learning_rate'],weight_decay=c['weight_decay'])
    fresh=not (root/'last.pt').exists()
    if not fresh:
        saved=torch.load(root/'last.pt',map_location='cuda',weights_only=True);assert saved['fingerprint']==fp
    else:
        source=Path(c['initial_resume_root']) if budget==50 else ROOT/'budget-50'
        if budget==40:assert json.loads((source/'completion.json').read_text())['tokens']==25_000_000
        checkpoint=source/'last.pt';source_sha=file_sha256(checkpoint)
        saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
        old_config=Path('configs/mistral-planner-full.json')
        expected=mf.fingerprint(json.loads(old_config.read_text()),old_config) if budget==50 else fp
        assert saved['fingerprint']==expected,'Source fingerprint changed; do not silently resume'
        lineage={'source_checkpoint':str(checkpoint),'source_sha256':source_sha,'source_tokens':saved['tokens'],
            'next_data_cursor':saved['position'],'original_fingerprint':saved['fingerprint'],'new_fingerprint':fp,
            'weights_optimizer_rng_preserved':True,'budget':budget,'target':c['phase_tokens'][str(budget)]}
        saved=migration_state(saved,c,budget)
        atomic_json(root/'lineage.json',lineage)
        if budget==50 and saved['selection'].get('pending_validation') is not None:
            check=f"check-{saved['tokens']:08d}"
            if (source/check).exists():shutil.copytree(source/check,root/check,dirs_exist_ok=True)
    p.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
    offsets=saved['offsets'];dual=DualBudget(**saved['dual']);seen=saved['tokens'];cursor=saved['position']
    elapsed=saved['elapsed'];selection=saved['selection'];origin=saved['schedule_origin_tokens'];last_save=seen
    if fresh and budget==40:
        offsets=pe.training_offsets(model,ctrl,p,q,corpus,rows,reserved[:4],.4)
    torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());del saved
    started=time.monotonic();measured=0.;new_tokens=0;torch.cuda.reset_peak_memory_stats()
    def save_current():save(root,p,opt,offsets,dual,seen,cursor,elapsed+time.monotonic()-started,fp,selection,origin)
    if fresh:save_current()
    def finish_validation():
        pe.save_weights(root,p,seen,fp)
        state=mf.validate(model,ctrl,p,q,offsets,budget/100,c,root/f'check-{seen:08d}')
        update_selection(selection,seen,state,c,budget)
        atomic_json(root/'selection.json',selection);save_current()
    if selection.get('pending_validation') is not None:
        assert selection['pending_validation']==seen
        finish_validation()
    while seen<selection['stop_at']:
        milestone=min([x for x in c['quality_checkpoint_tokens'] if x>seen]+[selection['stop_at']])
        length=min(512,milestone-seen);ids=pe.ids_for(corpus,rows,order[cursor],length)
        lr,temperature,mode=schedule(c,budget,seen,origin)
        for group in opt.param_groups:group['lr']=lr
        torch.cuda.synchronize();tick=time.monotonic()
        metrics=end_to_end_batch(model,ctrl,p,opt,q,ids,offsets,dual,mode=mode,temperature=temperature,
            beta=c['beta'],epsilon=c['epsilon'],chunk_size=c['teacher_head_chunk_size'])
        torch.cuda.synchronize();measured+=time.monotonic()-tick;seen+=length;new_tokens+=length;cursor+=1
        record={'stage':'e2e','budget':budget,'phase_tokens':seen,'lineage_e2e_tokens':seen+(25_000_000 if budget==40 else 0),
            'target_phase_tokens':selection['stop_at'],'next_cursor':cursor,'tokens_per_second':new_tokens/measured,
            'seconds':elapsed+time.monotonic()-started,'learning_rate':lr,'temperature':temperature,'gate_mode':mode,'metrics':metrics}
        if cursor%20==0 or seen==milestone:
            atomic_json(root/'progress.json',record);print(json.dumps(record),flush=True)
            with (root/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        pause=pe.PAUSE or (ROOT/'PAUSE').exists()
        if seen==milestone:selection['pending_validation']=seen
        if seen-last_save>=c['checkpoint_every_tokens'] or seen==milestone or pause:
            save_current();last_save=seen
        if pause:raise SystemExit(75)
        if seen==milestone:finish_validation()
    pe.save_weights(root,p,seen,fp)
    atomic_json(root/'completion.json',{'status':'complete','tokens':seen,'next_cursor':cursor,
        'lineage_e2e_tokens':seen+(25_000_000 if budget==40 else 0),'selection':selection,
        'seconds':elapsed+time.monotonic()-started,'fingerprint':fp,'offsets':offsets.cpu().tolist(),
        'backbone_frozen':all(not v.requires_grad and v.grad is None for v in model.parameters()),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated()})


@torch.no_grad()
def evaluate(c,budget):
    root=ROOT/f'budget-{budget}'
    if (root/'evaluation.json').exists():return
    done=json.loads((root/'completion.json').read_text());tokens=done['tokens']
    model,ctrl=mp.load_model(c);p=mp.make_planner(c);pe.load_stage_weights(p,root,tokens);p.eval()
    q=load_file(str(mf.PILOT/'thresholds.safetensors'),device='cuda')['q75']
    state=json.loads((root/f'check-{tokens:08d}'/'validation.json').read_text())
    tensors,records=data(c['heldout_corpus']);test=[r for r in records if r['split']=='test']
    overall,documents=mf.score_documents(model,ctrl,p,q,torch.tensor(state['offsets'],device='cuda'),
        tensors['input_ids'],test,budget/100,c)
    atomic_json(root/'evaluation-documents.json',documents)
    atomic_json(root/'evaluation.json',{'overall':overall,'selected_tokens':tokens,'lineage_e2e_tokens':done['lineage_e2e_tokens'],
        'domains':{d:aggregate([r for r in documents if r['domain']==d]) for d in ('web','code','conversation')},
        'selection':'Final phase checkpoint; threshold calibrated on separate validation; phase40 inherits final phase50'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['audit','e2e','evaluate']);parser.add_argument('--budget',type=int,choices=[50,40],default=50)
    args=parser.parse_args();c=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    mf.ROOT=ROOT
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGTERM,pe.request_pause);signal.signal(signal.SIGUSR1,pe.request_pause)
    if args.stage=='audit':full_data(c)
    elif args.stage=='evaluate':evaluate(c,args.budget)
    else:train(c,args.budget)


if __name__=='__main__':main()
