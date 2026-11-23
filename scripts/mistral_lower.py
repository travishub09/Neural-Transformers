"""Protected 40% checkpoint -> deterministic 30% -> deterministic 20%."""
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

ROOT=Path('artifacts/mistral-lower-v1')
CONFIG=Path('configs/mistral-lower-budgets.json')


def full_data(c):
    from scripts import mistral_phased as prior
    old=json.loads(Path('configs/mistral-planner-phased.json').read_text())
    prior.ROOT=ROOT/'source-data-audit';mf.ROOT=ROOT/'data-audit-base'
    ids,rows,reserved,order=prior.full_data(old)
    previous_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    extra,metadata=data(c['additional_lower_corpus']);_,heldout=data(c['heldout_corpus'])
    assert not {r['document_sha256'] for r in metadata}&{r['document_sha256'] for r in rows+heldout}
    extension=[dict(r,sequence_id=r['sequence_id']+len(ids)) for r in metadata]
    indices=[i for i,r in enumerate(extension) if r['split']=='train' and r['length']>=512]
    indices.sort(key=lambda i:hashlib.sha256((str(c['seed'])+':'+extension[i]['document_sha256']+':'+str(extension[i]['document_token_offset'])).encode()).hexdigest())
    order=order+[i+len(rows) for i in indices];rows+=extension;ids=torch.cat((ids,extra['input_ids']))
    source_done=json.loads((Path(c['source_40'])/'completion.json').read_text())
    needed=source_done['next_cursor']+sum(math.ceil(n/512) for n in c['phase_tokens'].values())+100
    assert len(order)>=needed,(len(order),needed)
    atomic_json(ROOT/'data-audit.json',{'available_full_windows':len(order),'required_windows':needed,
        'prior_order_sha256':previous_hash,'order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        'old_order_preserved':True,'new_document_hashes_disjoint':True,'continues_source_data_cursor':True})
    return ids,rows,reserved,order


def fingerprint(c):
    from scripts import mistral_phased as prior
    old=json.loads(Path('configs/mistral-planner-phased.json').read_text())
    result=prior.fingerprint(old)
    for path in [Path(__file__),CONFIG,Path(c['additional_lower_corpus'])/'manifest.json']:
        result[str(path)]=file_sha256(path)
    return result


def protect_source(c):
    source=Path(c['source_40']);target=ROOT/'protected-40';target.mkdir(parents=True,exist_ok=True)
    manifest=target/'preservation.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        for f in record['files']:assert file_sha256(target/f['name'])==f['sha256']
        return target
    done=json.loads((source/'completion.json').read_text());assert done['tokens']==15_000_000 and done['lineage_e2e_tokens']==40_000_000
    names=['last.pt','completion.json',f"checkpoint-{done['tokens']:08d}.json"]
    meta=json.loads((source/names[-1]).read_text());names.append(meta['file']['file'])
    record={'source':str(source),'source_lineage_e2e_tokens':40_000_000,'files':[]}
    for name in names:
        f=source/name;dest=target/name;digest=file_sha256(f)
        if not dest.exists():shutil.copy2(f,dest)
        assert file_sha256(dest)==digest
        dest.chmod(0o444)
        record['files'].append({'name':name,'sha256':digest,'bytes':dest.stat().st_size})
    atomic_json(manifest,record)
    return target


def migration_state(saved,c,budget):
    migrated=dict(saved)
    migrated.update(tokens=0,elapsed=0.,schedule_origin_tokens=0)
    migrated['selection']={'curve':[],'stop_at':c['phase_tokens'][str(budget)],'hard_tail':False}
    migrated['dual']=asdict(DualBudget(budget/100,c['alpha_initial'],c['dual_learning_rate'],c['maximum_alpha']))
    return migrated


def schedule(c,budget,seen,origin=0):
    total=c['phase_tokens'][str(budget)]
    factor=min(1.,(seen+512)/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/total)))
    return c['quality_learning_rate']*factor,.1,'hard'


def update_selection(s,seen,state,c,budget):
    kl=state['overall']['teacher_kl_nats']
    s['curve'].append({'tokens':seen,'validation_kl':kl,'validation_ppl_increase':state['overall']['relative_ppl_increase'],'validation_kept':state['overall']['routed_kept']})
    if seen==c['minimum_phase_tokens']:
        previous=[x['validation_kl'] for x in s['curve'] if x['tokens']<seen]
        improvement=min(previous)-kl if previous else float('inf')
        extend=improvement>=c['extension_kl_improvement']
        s['extension_decision']={'at_tokens':seen,'improvement':improvement,'extend_to_10M':extend}
        if not extend:s['stop_at']=seen
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
        from scripts import mistral_phased as prior
        source=protect_source(c) if budget==30 else ROOT/'budget-30'
        checkpoint=source/'last.pt';source_sha=file_sha256(checkpoint)
        source_done=json.loads((source/'completion.json').read_text())
        saved=torch.load(checkpoint,map_location='cuda',weights_only=True)
        expected=prior.fingerprint(json.loads(Path('configs/mistral-planner-phased.json').read_text())) if budget==30 else fp
        assert saved['fingerprint']==expected,'Producing code changed; refuse silent migration'
        lineage={'source_checkpoint':str(checkpoint),'source_sha256':source_sha,'source_tokens':saved['tokens'],
            'next_data_cursor':saved['position'],'original_fingerprint':saved['fingerprint'],'new_fingerprint':fp,
            'weights_optimizer_rng_preserved':True,'budget':budget,'target':c['phase_tokens'][str(budget)],
            'source_lineage_e2e_tokens':source_done['lineage_e2e_tokens']}
        saved=migration_state(saved,c,budget);atomic_json(root/'lineage.json',lineage)
    p.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
    offsets=saved['offsets'];dual=DualBudget(**saved['dual']);seen=saved['tokens'];cursor=saved['position']
    elapsed=saved['elapsed'];selection=saved['selection'];origin=saved['schedule_origin_tokens'];last_save=seen
    lineage=json.loads((root/'lineage.json').read_text())
    if fresh:
        offsets=pe.training_offsets(model,ctrl,p,q,corpus,rows,reserved[:4],budget/100)
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
        record={'stage':'e2e','budget':budget,'phase_tokens':seen,'lineage_e2e_tokens':seen+lineage['source_lineage_e2e_tokens'],
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
        'lineage_e2e_tokens':seen+lineage['source_lineage_e2e_tokens'],'selection':selection,
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
        'selection':'Final phase checkpoint; threshold calibrated on separate validation; 30% inherits protected final 40%; 20% inherits final 30%'})


@torch.no_grad()
def evaluate25(c):
    root=ROOT/'budget-30';done=json.loads((root/'completion.json').read_text());tokens=done['tokens']
    out=ROOT/'calibrated-25-from-30'
    if (out/'evaluation.json').exists():return
    model,ctrl=mp.load_model(c);p=mp.make_planner(c);pe.load_stage_weights(p,root,tokens);p.eval()
    q=load_file(str(mf.PILOT/'thresholds.safetensors'),device='cuda')['q75']
    base=torch.tensor(done['offsets'],device='cuda')
    state=mf.validate(model,ctrl,p,q,base,.25,c,out)
    tensors,records=data(c['heldout_corpus']);rows=[r for r in records if r['split']=='test']
    overall,documents=mf.score_documents(model,ctrl,p,q,torch.tensor(state['offsets'],device='cuda'),tensors['input_ids'],rows,.25,c)
    atomic_json(out/'evaluation-documents.json',documents)
    atomic_json(out/'evaluation.json',{'overall':overall,'source_budget':30,'target_budget':25,'trained_at_25':False,
        'domains':{d:aggregate([r for r in documents if r['domain']==d]) for d in ('web','code','conversation')}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['protect','audit','e2e','evaluate','evaluate25']);parser.add_argument('--budget',type=int,choices=[30,20],default=30)
    args=parser.parse_args();c=json.loads(CONFIG.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    mf.ROOT=ROOT
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGTERM,pe.request_pause);signal.signal(signal.SIGUSR1,pe.request_pause)
    if args.stage=='protect':protect_source(c)
    elif args.stage=='audit':full_data(c)
    elif args.stage=='evaluate':evaluate(c,args.budget)
    elif args.stage=='evaluate25':evaluate25(c)
    else:train(c,args.budget)


if __name__=='__main__':main()
