"""Original upfront planner, native Mistral backbone, bounded scaling pilot."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import torch
from transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from pathpred.collection import atomic_json,atomic_tensors,file_sha256
from pathpred.gated_qwen import gate_mistral
from pathpred.open_loop import PlannerConfig,SuffixPlanner
from pathpred.planner_training import copy_batch,end_to_end_batch
from pathpred.planner_decode import decode_batch
from pathpred.end_to_end import DualBudget
from pathpred.budget_calibration import calibrate_threshold
from scripts import planner_experiment as pe
from scripts.neuron_guessing import data
from scripts.oracle_quality import aggregate,bootstrap_quality

ROOT=Path('artifacts/mistral-pilot-v1')


def load_model(c):
    path=Path(c['model_path'])
    if not (path/'model.safetensors.index.json').exists():
        snapshot_download(c['model_id'],revision=c['revision'],local_dir=str(path),
            allow_patterns=['*.safetensors','*.json','tokenizer.model'],max_workers=4)
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='eager',local_files_only=True).eval()
    assert (model.config.num_hidden_layers,model.config.intermediate_size,model.config.hidden_size)==(32,14336,4096)
    return model,gate_mistral(model)


def make_planner(c):
    torch.manual_seed(c['seed'])
    return SuffixPlanner(PlannerConfig(llm_layers=32,units=14336,hidden_size=4096,
        width=c['width'],depth=c['depth'],heads=c['heads'],dense_prefix=2,autoregressive=True,previous_path=True)).cuda()


def training_order(c):
    tensors,rows=data(c['train_corpus']);_,heldout=data(c['heldout_corpus'])
    forbidden={r['document_sha256'] for r in heldout}
    order=[i for i,r in enumerate(rows) if r['split']=='train' and r['length']>=512 and r['document_sha256'] not in forbidden]
    order.sort(key=lambda i:hashlib.sha256((str(c['seed'])+':'+rows[i]['document_sha256']+':'+str(rows[i]['document_token_offset'])).encode()).hexdigest())
    reserved,order=order[:16],order[16:]
    assert len(order)>=sum(math.ceil(c[s+'_tokens']/512) for s in ('copy','scheduled','e2e'))
    atomic_json(ROOT/'data-audit.json',{'train_windows':len(order),'reserved_calibration_windows':reserved,
        'order_sha256':hashlib.sha256(json.dumps(order).encode()).hexdigest(),'heldout_hashes_excluded':True,
        'model_revision':c['revision'],'tokenizer':'Native Mistral token IDs; no Qwen IDs used'})
    return tensors['input_ids'],rows,reserved,order


def fingerprint(config,c):
    files=list(Path('pathpred').glob('*.py'))+[config,Path(__file__),Path('scripts/planner_experiment.py'),
        Path(c['train_corpus'])/'manifest.json',Path(c['heldout_corpus'])/'manifest.json',ROOT/'thresholds.safetensors']
    return {str(p):file_sha256(p) for p in sorted(files)}


@torch.no_grad()
def prepare(c):
    target=ROOT/'thresholds.safetensors'
    if target.exists():return
    model,ctrl=load_model(c);corpus,rows,reserved,order=training_order(c)
    # Validate the full-size unmodified model as well as the CPU unit tests.
    ids=pe.ids_for(corpus,rows,reserved[0],16)
    ctrl.remove();dense=model(ids).logits
    ctrl=gate_mistral(model)
    with ctrl.use_gates({(l,'mlp'):torch.ones(14336,device='cuda') for l in range(32)}):
        gated=model(ids).logits
    difference=float((dense-gated).abs().max());assert difference==0.,difference
    samples={l:[] for l in range(32)};handles=[]
    for l,layer in enumerate(model.model.layers):
        def observe(module,args,index=l):samples[index].append(args[0][:,::8].detach().abs().to('cpu',torch.float16).reshape(-1,14336))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(observe))
    try:
        for number,index in enumerate(reserved):
            model.model(pe.ids_for(corpus,rows,index),use_cache=False)
            print(json.dumps({'stage':'calibration_collection','window':number+1,'total':len(reserved)}),flush=True)
    finally:
        for handle in handles:handle.remove()
    q=[];frequency={50:[],40:[]}
    for l in range(32):
        x=torch.cat(samples.pop(l)).cuda().float();q.append(x.quantile(.75,dim=0).cpu())
        for budget in (50,40):
            top=x.topk(round(14336*budget/100),dim=-1).indices
            freq=torch.zeros_like(x,dtype=torch.bool).scatter_(-1,top,True).float().mean(0)
            frequency[budget].append(freq.cpu())
        del x
    value={'q75':torch.stack(q),'frequency_50':torch.stack(frequency[50]),'frequency_40':torch.stack(frequency[40])}
    entry=atomic_tensors(target,value)
    p=make_planner(c)
    atomic_json(ROOT/'preflight.json',{'model_revision':c['revision'],'gate_one_max_logit_difference':difference,
        'parameters':sum(v.numel() for v in model.parameters()),'planner':p.accounting(),
        'planner_parameter_fraction':sum(v.numel() for v in p.parameters())/sum(v.numel() for v in model.parameters()),
        'threshold_file':entry,'threshold_positions_per_layer':1024,'threshold_data':'16 reserved train windows, disjoint all training stages and held-out',
        'attention_dense':True,'sequence_prior':False,'previous_token_path':True,
        'physical_sparse_kernels':False})


def initialize_copy(p,c):
    cal=load_file(str(ROOT/'thresholds.safetensors'),device='cuda')
    prior=((cal['frequency_50']+cal['frequency_40'])/2)[2:].clamp(.001,.999)
    with torch.no_grad():
        p.output_bias.copy_((c['positive_weight']*prior/(1-prior)).log());p.output_weight.mul_(.01)


def smoke(c):
    model,ctrl=load_model(c);q=load_file(str(ROOT/'thresholds.safetensors'),device='cuda')['q75']
    p=make_planner(c);initialize_copy(p,c)
    corpus,rows,reserved,order=training_order(c);ids=pe.ids_for(corpus,rows,order[0]);result={}
    optimizer=torch.optim.AdamW(p.parameters(),lr=c['quality_learning_rate'])
    for stage,count in [('copy',4),('scheduled',4),('e2e',6)]:
        elapsed=[];torch.cuda.reset_peak_memory_stats()
        for i in range(count):
            torch.cuda.synchronize();start=time.monotonic()
            if stage=='e2e':metrics=end_to_end_batch(model,ctrl,p,optimizer,q,ids,torch.zeros(p.steps,device='cuda'),
                DualBudget(.5,.1,.05,100.),mode='hard' if i==count-1 else 'gumbel',chunk_size=32)
            else:metrics=copy_batch(model,ctrl,p,optimizer,q,ids,keep=.5,scheduled=stage=='scheduled',teacher_probability=.5)
            torch.cuda.synchronize();elapsed.append(time.monotonic()-start)
        result[stage]={'seconds_per_step':elapsed,'steady_tokens_per_second':512*(count-1)/sum(elapsed[1:]),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'metrics':metrics}
        atomic_json(ROOT/'gpu-smoke.json',result);print(json.dumps({'stage':stage,**result[stage]}),flush=True)
    estimate=sum(c[s+'_tokens']/result[s]['steady_tokens_per_second'] for s in result)
    full=c['proposed_full_run'];full_seconds=(full['copy_tokens']/result['copy']['steady_tokens_per_second']+
        full['scheduled_tokens']/result['scheduled']['steady_tokens_per_second']+
        2*full['e2e_tokens_per_budget']/result['e2e']['steady_tokens_per_second'])
    atomic_json(ROOT/'runtime-estimate.json',{'pilot_training_seconds':estimate,'full_training_seconds':full_seconds,
        'excludes_download_calibration_evaluation':True,'gpu':torch.cuda.get_device_name()})


def train(c,config,stage):
    root=ROOT/stage;root.mkdir(exist_ok=True)
    if (root/'completion.json').exists():return
    fp=fingerprint(config,c);model,ctrl=load_model(c)
    q=load_file(str(ROOT/'thresholds.safetensors'),device='cuda')['q75'];p=make_planner(c)
    corpus,rows,reserved,order=training_order(c)
    if stage=='copy':initialize_copy(p,c)
    else:
        previous='copy' if stage=='scheduled' else 'scheduled'
        pe.load_stage_weights(p,ROOT/previous,c[previous+'_tokens'])
    skip=0 if stage=='copy' else math.ceil(c['copy_tokens']/512)
    if stage=='e2e':skip+=math.ceil(c['scheduled_tokens']/512)
    order=order[skip:]
    quality=stage=='e2e';total=c[stage+'_tokens'];lr=c['quality_learning_rate'] if quality else c['copy_learning_rate']
    optimizer=torch.optim.AdamW(p.parameters(),lr=lr,weight_decay=c['weight_decay'])
    offsets=torch.zeros(p.steps,device='cuda');dual=DualBudget(c['budget'],c['alpha_initial'],c['dual_learning_rate'],c['maximum_alpha']) if quality else None
    seen=position=last_save=0;elapsed=0.
    if (root/'last.pt').exists():
        saved=torch.load(root/'last.pt',map_location='cuda',weights_only=True);assert saved['fingerprint']==fp
        p.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);offsets=saved['offsets']
        dual=DualBudget(**saved['dual']) if quality else None;seen=saved['tokens'];position=saved['position'];elapsed=saved['elapsed'];last_save=seen
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());del saved
    elif quality:offsets=pe.training_offsets(model,ctrl,p,q,corpus,rows,reserved[:4],c['budget'])
    start=time.monotonic();measured=0.;new_tokens=0;torch.cuda.reset_peak_memory_stats()
    for index in range(position,len(order)):
        if seen>=total:break
        length=min(512,total-seen);ids=pe.ids_for(corpus,rows,order[index],length)
        factor=min(1.,(seen+length)/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/total)))
        for group in optimizer.param_groups:group['lr']=lr*factor
        torch.cuda.synchronize();tick=time.monotonic()
        if quality:
            fraction=seen/total;mode='hard' if fraction>=1-c['hard_fraction'] else 'gumbel'
            temperature=c['temperatures'][0]*(c['temperatures'][1]/c['temperatures'][0])**min(1.,fraction/(1-c['hard_fraction']))
            metrics=end_to_end_batch(model,ctrl,p,optimizer,q,ids,offsets,dual,mode=mode,temperature=temperature,
                beta=c['beta'],epsilon=c['epsilon'],chunk_size=c['teacher_head_chunk_size'])
        else:metrics=copy_batch(model,ctrl,p,optimizer,q,ids,keep=.5 if index%2==0 else .4,
            scheduled=stage=='scheduled',teacher_probability=max(0.,1-seen/total),positive_weight=c['positive_weight'])
        torch.cuda.synchronize();measured+=time.monotonic()-tick;seen+=length;new_tokens+=length
        record={'stage':stage,'tokens':seen,'target_tokens':total,'tokens_per_second':new_tokens/measured,
            'seconds':elapsed+time.monotonic()-start,'metrics':metrics}
        if index%20==0 or seen==total:
            atomic_json(root/'progress.json',record);print(json.dumps(record),flush=True)
            with (root/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        pause=pe.PAUSE or (ROOT/'PAUSE').exists()
        if seen-last_save>=c['checkpoint_every_tokens'] or seen==total or pause:
            pe.save_resume(root,p,optimizer,offsets,dual,seen,index+1,elapsed+time.monotonic()-start,fp);last_save=seen
        if pause:raise SystemExit(75)
    assert seen==total
    pe.save_weights(root,p,seen,fp)
    atomic_json(root/'completion.json',{'status':'complete','tokens':seen,'seconds':elapsed+time.monotonic()-start,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'fingerprint':fp,'offsets':offsets.cpu().tolist(),
        'backbone_frozen':all(not v.requires_grad and v.grad is None for v in model.parameters())})


def batches(corpus,rows,size):
    for start in range(0,len(rows),size):
        selected=rows[start:start+size];lengths=torch.tensor([r['length'] for r in selected],device='cuda')
        ids=torch.full((len(selected),int(lengths.max())),2,dtype=torch.long,device='cuda')
        for i,r in enumerate(selected):ids[i,:r['length']]=corpus[r['sequence_id'],:r['length']].cuda().long()
        yield selected,ids,lengths


def progress(token,total):
    if pe.PAUSE or (ROOT/'PAUSE').exists():raise SystemExit(75)
    print(json.dumps({'stage':'cached_evaluation','token':token,'total':total}),flush=True)


@torch.no_grad()
def evaluate(c):
    if (ROOT/'evaluation.json').exists():return
    model,ctrl=load_model(c);p=make_planner(c);pe.load_stage_weights(p,ROOT/'e2e',c['e2e_tokens']);p.eval()
    q=load_file(str(ROOT/'thresholds.safetensors'),device='cuda')['q75']
    offsets=torch.tensor(json.loads((ROOT/'e2e/completion.json').read_text())['offsets'],device='cuda')
    corpus,records=data(c['heldout_corpus']);corpus=corpus['input_ids']
    validation=[r for r in records if r['split']=='validation'];test=[r for r in records if r['split']=='test']
    trials_path=ROOT/'calibration-trials.json';trials=json.loads(trials_path.read_text()) if trials_path.exists() else {}
    def measure(shift):
        key=repr(float(shift))
        if key in trials:return trials[key]
        count=kept=0.
        for _,ids,lengths in batches(corpus,validation,c['eval_batch_size']):
            result=decode_batch(model,ctrl,p,q,ids,lengths,offsets=offsets+shift,keep=c['budget'],quality=False,progress=progress)
            count+=float(result['sums'][:,4].sum());kept+=float(result['kept_sum'].sum())
        trials[key]=kept/count;atomic_json(trials_path,trials);return trials[key]
    if (ROOT/'threshold.json').exists():cal=json.loads((ROOT/'threshold.json').read_text())
    else:
        cal=calibrate_threshold(measure,c['budget'],tolerance=c['calibration_tolerance'],max_evaluations=c['calibration_max_evaluations'])
        cal['offsets']=(offsets+cal['shift']).cpu().tolist();atomic_json(ROOT/'threshold.json',cal)
    offsets=torch.tensor(cal['offsets'],device='cuda');rows=[]
    for metadata,ids,lengths in batches(corpus,test,c['eval_batch_size']):
        result=decode_batch(model,ctrl,p,q,ids,lengths,offsets=offsets,keep=c['budget'],progress=progress)
        for i,r in enumerate(metadata):
            values=result['sums'][i].tolist();row=dict(zip(('teacher_nll','student_nll','kl','top1_agreement','tokens'),values))
            row.update(document=r['document_sha256'],domain=r['domain'],routed_kept=float(result['kept_sum'][i]/values[-1]));rows.append(row)
    overall=aggregate(rows);overall.update(bootstrap_quality(rows,2000,c['seed']))
    overall['routed_kept']=sum(r['routed_kept']*r['tokens'] for r in rows)/overall['tokens']
    atomic_json(ROOT/'evaluation-documents.json',rows)
    atomic_json(ROOT/'evaluation.json',{'overall':overall,'domains':{d:aggregate([r for r in rows if r['domain']==d]) for d in ('web','code','conversation')},
        'pilot_not_full_run':True,'training_tokens':{s:c[s+'_tokens'] for s in ('copy','scheduled','e2e')},
        'context':'Original upfront AR planner, no sequence prior, exact cached previous-token paths; full held-out text spans retokenized for Mistral'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/mistral-planner-pilot.json'))
    parser.add_argument('stage',choices=['prepare','smoke','copy','scheduled','e2e','evaluate']);args=parser.parse_args()
    c=json.loads(args.config.read_text());ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGTERM,pe.request_pause);signal.signal(signal.SIGUSR1,pe.request_pause)
    if args.stage=='prepare':prepare(c)
    elif args.stage=='smoke':smoke(c)
    elif args.stage=='evaluate':evaluate(c)
    else:train(c,args.config,args.stage)


if __name__=='__main__':main()
