"""Two fixed-budget, hard-threshold continuations; 20M presentations total."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.end_to_end import DualBudget
from pathpred.neuron_prediction import NeuronPredictor, neuron_config
from scripts.neuron_guessing import backbone, data, CAL
from scripts.calibrate_native import selections, calibrate, evaluate
from scripts import e2e_pilot

ROOT=Path('artifacts/e2e-full-v1')
SIGNAL='binary'
PILOT_ROOT=Path('artifacts/e2e-pilot-v1')
CALIBRATION_ROOT=Path('artifacts/native-calibration-v1')


def config(): return json.loads(Path('configs/e2e-full.json').read_text())


def prepare_training():
    corpus,records=data('data/e2e-full-corpus-v1')
    forbidden=set()
    for name in ['neuron-training-corpus-v1','stage3-corpus-v1']:
        _,old=data('data/'+name)
        forbidden.update(r['document_sha256'] for r in old)
    order=[i for i,r in enumerate(records) if r['split']=='train' and r['length']>=512
           and r['document_sha256'] not in forbidden]
    seed=str(config()['seed'])
    order.sort(key=lambda i:hashlib.sha256((seed+records[i]['document_sha256']+
                   ':'+str(records[i]['document_token_offset'])).encode()).hexdigest())
    boundaries=[0]+config()['milestones']
    required=sum(math.ceil((b-a)/512) for a,b in zip(boundaries,boundaries[1:]))
    if len(order)<required:raise ValueError(f'Not enough fresh full windows: {len(order)} < {required}')
    order=order[:required]
    audit={'full_windows':len(order),'unique_documents':len({records[i]['document_sha256'] for i in order}),
           'excluded_document_hashes':len(forbidden),'prior_training_validation_test_documents_disjoint':True,
           'window_indices':order,'same_order_both_branches':True,
           'per_branch_presentations':config()['train_tokens'],
           'total_presentations':2*config()['train_tokens'],
           'note':'Fresh documents relative to the previous corpora; each branch sees the same 10M tokens once.'}
    path=ROOT/'data-audit.json'
    if path.exists():assert json.loads(path.read_text())==audit
    else:atomic_json(path,audit)
    return corpus['input_ids'],records,order


def fingerprint():
    names=['configs/e2e-full.json','configs/e2e-full-corpus.json','docs/e2e-full-protocol.md',
           'scripts/e2e_full.py','scripts/e2e_pilot.py','scripts/calibrate_native.py',
           'pathpred/end_to_end.py','pathpred/budget_calibration.py','pathpred/neuron_prediction.py',
           'pathpred/path_model.py','pathpred/gated_qwen.py','pathpred/reordering.py',
           'data/e2e-full-corpus-v1/manifest.json','data/stage3-corpus-v1/manifest.json',
           str(CALIBRATION_ROOT/'threshold-50.json'),
           str(CALIBRATION_ROOT/'threshold-40.json')]
    return {n:file_sha256(Path(n)) for n in names}


def save_resume(root,p,optimizer,offsets,dual,tokens,position,elapsed,fp):
    temporary=root/'last.tmp'
    torch.save({'predictor':p.state_dict(),'optimizer':optimizer.state_dict(),'offsets':offsets,
                'dual':asdict(dual),'tokens':tokens,'position':position,'elapsed':elapsed,'fingerprint':fp,
                'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},temporary)
    temporary.replace(root/'last.pt')


def checkpoint(model,controller,p,q,root,tokens,offsets,dual,fp,eval_corpus,validation,test):
    label=f'{tokens:08d}';meta_path=root/f'checkpoint-{label}.json'
    if not meta_path.exists():
        entry=atomic_tensors(root/f'predictor-{label}.safetensors',
                            {k:v.detach().cpu().contiguous() for k,v in p.state_dict().items()})
        atomic_json(meta_path,{'tokens':tokens,'target':dual.target,'file':entry,'dual':asdict(dual),
                              'training_offsets':offsets.cpu().tolist(),'fingerprint':fp})
    else:
        entry=json.loads(meta_path.read_text())['file']
        assert file_sha256(root/entry['file'])==entry['sha256']
    # Save native training-policy performance too; threshold calibration never
    # optimizes quality. It does not change subsequent training offsets or alpha.
    native_path=root/f'native-{label}.json'
    threshold_path=root/f'threshold-{label}.json'
    if threshold_path.exists():
        calibrated=torch.tensor(json.loads(threshold_path.read_text())['offsets'],device='cuda')
    else:
        calibrated,_=calibrate(model,controller,p,q,offsets,eval_corpus,validation,dual.target,threshold_path)
    if not native_path.exists():
        evaluate(model,controller,p,q,offsets,eval_corpus,test,native_path)
    matched_path=root/f'evaluation-{label}.json'
    if not matched_path.exists():
        evaluate(model,controller,p,q,calibrated,eval_corpus,test,matched_path)


def run(smoke=False):
    c=config();e2e_pilot.config=lambda:c
    corpus,records,order=prepare_training();fp=fingerprint()
    eval_corpus,validation,test=selections()
    model,controller=backbone();q=load_file(str(CAL),device='cuda')['q75']
    total_start=time.monotonic()
    for budget,label in [(50,'2000000'),(40,'3000000')]:
        root=ROOT/f'budget-{budget}';root.mkdir(parents=True,exist_ok=True)
        if (root/'training.json').exists() and not smoke:
            assert json.loads((root/'training.json').read_text())['fingerprint']==fp
            continue
        torch.manual_seed(c['seed'])
        p=NeuronPredictor(neuron_config(SIGNAL)).cuda().eval()
        warm=PILOT_ROOT/f'predictor-{label}.safetensors'
        calibration=json.loads((CALIBRATION_ROOT/f'threshold-{budget}.json').read_text())
        assert file_sha256(warm)==calibration['checkpoint_sha256']
        p.load_state_dict(load_file(str(warm),device='cuda'))
        offsets=torch.tensor(calibration['offsets'],device='cuda')
        pilot_meta=json.loads((PILOT_ROOT/f'checkpoint-{label}.json').read_text())
        dual=DualBudget(budget/100,pilot_meta['dual']['alpha'],c['dual_learning_rate'],c['maximum_alpha'])
        optimizer=torch.optim.AdamW(p.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        seen=position=0;elapsed=0.;last_save=0
        if (root/'last.pt').exists() and not smoke:
            state=torch.load(root/'last.pt',map_location='cuda',weights_only=True)
            assert state['fingerprint']==fp
            p.load_state_dict(state['predictor']);optimizer.load_state_dict(state['optimizer'])
            offsets=state['offsets'];dual=DualBudget(**state['dual']);seen=state['tokens'];position=state['position']
            elapsed=state['elapsed'];last_save=seen
            torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu());del state
        if seen in c['milestones'] and not smoke:
            checkpoint(model,controller,p,q,root,seen,offsets,dual,fp,eval_corpus,validation,test)
        start=time.monotonic();step_time=0.;new_tokens=0;torch.cuda.reset_peak_memory_stats()
        for index in range(position,len(order)):
            if seen>=c['train_tokens']:break
            boundary=min(n for n in c['milestones'] if n>seen)
            length=min(512,boundary-seen)
            r=records[order[index]]
            ids=corpus[r['sequence_id']:r['sequence_id']+1,:length].long().cuda()
            factor=min(1.,(seen+length)/c['warmup_tokens'])*(.1+.45*(1+math.cos(math.pi*seen/c['train_tokens'])))
            temperature=.3*(.1/.3)**min(1.,seen/2000000)
            torch.cuda.synchronize();tick=time.monotonic()
            metrics=e2e_pilot.train_batch(model,controller,p,optimizer,q,offsets,ids,dual,
                                         temperature,'hard',c['learning_rate']*(1. if smoke else factor))
            torch.cuda.synchronize();duration=time.monotonic()-tick
            seen+=length;new_tokens+=length;step_time+=duration;position=index+1
            progress={'budget':budget,'tokens':seen,'mode':'hard','temperature':temperature,
                      'tokens_per_second':new_tokens/step_time,'elapsed_seconds':elapsed+time.monotonic()-start,
                      'metrics':metrics,'position':position}
            if index%20==0 or smoke:
                atomic_json(root/('smoke-progress.json' if smoke else 'progress.json'),progress)
                print(json.dumps(progress),flush=True)
            if smoke and index==7:
                atomic_json(ROOT/'smoke.json',{'status':'complete','steps':8,'tokens':new_tokens,
                    'tokens_per_second':new_tokens/step_time,'metrics':metrics,
                    'backbone_frozen':all(v.grad is None and not v.requires_grad for v in model.parameters()),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'fingerprint':fp})
                return
            guard=time.monotonic()-total_start>c['maximum_training_hours']*3600
            if seen-last_save>=c['checkpoint_every_tokens'] or seen in c['milestones'] or guard:
                save_resume(root,p,optimizer,offsets,dual,seen,position,elapsed+time.monotonic()-start,fp)
                last_save=seen
            if guard:raise TimeoutError('Eight-hour combined training guard; latest state persisted')
            if seen in c['milestones']:
                checkpoint(model,controller,p,q,root,seen,offsets,dual,fp,eval_corpus,validation,test)
        assert seen==c['train_tokens']
        atomic_json(root/'training.json',{'status':'complete','tokens':seen,'gradient_update_windows':position,
            'seconds':elapsed+time.monotonic()-start,'fingerprint':fp,'target':budget/100,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'backbone_frozen':all(v.grad is None and not v.requires_grad for v in model.parameters())})
        del optimizer,p;torch.cuda.empty_cache()


def main():
    global ROOT,SIGNAL,PILOT_ROOT,CALIBRATION_ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--signal',choices=['binary','hidden'],default='binary')
    a=parser.parse_args();SIGNAL=a.signal
    if SIGNAL=='hidden':
        ROOT=Path('artifacts/e2e-hidden-full-v1')
        PILOT_ROOT=Path('artifacts/e2e-hidden-pilot-v1')
        CALIBRATION_ROOT=Path('artifacts/hidden-native-calibration-v1')
    ROOT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:run(a.smoke)
    except BaseException as e:
        atomic_json(ROOT/'failure.json',{'status':'failed','error_type':type(e).__name__,'error':str(e)})
        raise
    else:
        if not a.smoke:atomic_json(ROOT/'completion.json',{'status':'complete','total_new_training_presentations':20000000})


if __name__=='__main__':main()
