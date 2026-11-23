"""Matched-data CE training and an explicit held-out centroid-accuracy gate."""
import argparse
import gc
import json
import math
from pathlib import Path
import signal
import time
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.neuroprefetcher import NeuroConfig, NeuroPredictor, qwen_overhead
from pathpred.neuro_features import capture_neuro, feature_tuple, target_ids
from pathpred.neuro_profile import runtime_overhead
from scripts.neuron_guessing import backbone, data
from scripts.neuro_data import load_matching, window_ids
from scripts.oracle_quality import select_documents
from scripts.calibrate_native import selections, ids_for


def load_banks(c, budget, device='cuda'):
    folder=Path(c['root'])/f'budget-{round(100*budget)}'
    manifest=json.loads((folder/'centroids.json').read_text()); banks={};thresholds={};aggregates={}
    for key, record in manifest['banks'].items():
        path=folder/(key+'.safetensors');assert file_sha256(path)==record['file']['sha256']
        loaded=load_file(str(path),device=device)
        banks[key]=loaded['masks'];thresholds[key]=loaded['threshold'];aggregates[key]=loaded['aggregate']
    return banks,thresholds,aggregates


def predictor(c, device='cuda'):
    return NeuroPredictor(NeuroConfig(width=c['width'],depth=c['depth'],centroids=c['centroids'],
                                     prefix=c['dense_prefix'],dropout=c['dropout'])).to(device)


def validation_data():
    corpus,_,test=selections();_,records=data('data/stage3-corpus-v1')
    validation=select_documents(records,'validation',8,{'seed':20261111,'minimum_sequence_length':128})
    assert not {r['document_sha256'] for r in validation}&{r['document_sha256'] for r in test}
    return corpus,validation,test


@torch.no_grad()
def accuracy(model,controller,p,banks,thresholds,corpus,selected,frequencies):
    p.eval();shape=(2,p.config.layers-p.config.prefix)
    correct=torch.zeros(shape,device='cuda',dtype=torch.long);majority=correct.clone();count=0
    totals=torch.zeros((*shape,p.config.centroids),device='cuda',dtype=torch.long)
    common=frequencies.argmax(-1)
    for row in selected:
        ids=ids_for(corpus,row)
        with controller.use_gates(),capture_neuro(model) as trace:model.model(ids,use_cache=False)
        context,labels=target_ids(trace,thresholds,banks)
        with torch.autocast('cuda',dtype=torch.bfloat16): logits=p(feature_tuple(trace),context)
        for branch,(scores,target) in enumerate(zip(logits,labels)):
            # Align with evaluated next-token positions, excluding final token.
            prediction=scores[:,:-1].argmax(-1);target=target[:,:-1]
            correct[branch]+=(prediction==target).sum((0,1))
            majority[branch]+=(common[branch]==target).sum((0,1))
            for layer in range(shape[1]): totals[branch,layer]+=torch.bincount(target[...,layer].flatten(),minlength=p.config.centroids)
        count+=ids.shape[1]-1
    assert count>0
    rate=correct.double()/count
    return {'tokens':count,'gate_top1':float(rate[0].mean()),'up_top1':float(rate[1].mean()),
            'per_layer_accuracy':rate.cpu().tolist(),'most_frequent_baseline':(majority.double()/count).cpu().tolist(),
            'occupied_label_classes':(totals>0).sum(-1).cpu().tolist(),
            'evaluation':'held-out validation documents; dense full-sequence feature pass; no optimizer updates'}


def identity(c,budget):
    names=['pathpred/neuroprefetcher.py','pathpred/neuro_features.py','scripts/neuro_data.py',
           'scripts/neuro_prepare.py','scripts/neuro_train.py','configs/neuroprefetcher.json',
           c['planner_config'],str(Path(c['root'])/'data-audit.json'),
           str(Path(c['root'])/f'budget-{round(100*budget)}'/'centroids.json')]
    return {name:file_sha256(Path(name)) for name in names}


def milestone_ready(root, milestone):
    return all((root/name).exists() for name in (f'accuracy-{milestone}.json',
        f'checkpoint-{milestone}.json',f'predictor-{milestone}.safetensors'))


def save_milestone(root,p,milestone,seen,fp,results):
    # Accuracy is the commit marker, written only after weights and metadata.
    entry=atomic_tensors(root/f'predictor-{milestone}.safetensors',
        {k:v.detach().cpu().contiguous() for k,v in p.state_dict().items()})
    atomic_json(root/f'checkpoint-{milestone}.json',{'tokens':seen,'file':entry,'fingerprint':fp})
    results.update(tokens_trained=seen,requested_milestone=milestone)
    atomic_json(root/f'accuracy-{milestone}.json',results)


def train(c,budget):
    root=Path(c['root'])/f'budget-{round(100*budget)}';root.mkdir(parents=True,exist_ok=True)
    sources,windows,audit=load_matching(c)
    assert json.loads((Path(c['root'])/'data-audit.json').read_text())==audit
    banks,thresholds,_=load_banks(c,budget);model,controller=backbone()
    torch.manual_seed(c['seed']);p=predictor(c)
    optimizer=torch.optim.AdamW(p.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    corpus,validation,_=validation_data();fp=identity(c,budget)
    seen=position=last_save=0;elapsed=0.;train_correct=torch.zeros(2,device='cuda');train_count=0
    frequencies=torch.zeros(2,26,c['centroids'],device='cuda',dtype=torch.long)
    if (root/'last.pt').exists():
        state=torch.load(root/'last.pt',map_location='cuda',weights_only=True);assert state['fingerprint']==fp
        p.load_state_dict(state['predictor']);optimizer.load_state_dict(state['optimizer'])
        seen=last_save=state['tokens'];position=state['position'];elapsed=state['elapsed']
        frequencies=state['frequencies'];torch.set_rng_state(state['cpu_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        del state
    started=time.monotonic();stop=[];signal.signal(signal.SIGTERM,lambda s,f:stop.append(s))
    atomic_json(root/'accounting.json',qwen_overhead(p,model))
    def save():
        path=root/'last.tmp';torch.save({'predictor':p.state_dict(),'optimizer':optimizer.state_dict(),
            'tokens':seen,'position':position,'elapsed':elapsed+time.monotonic()-started,'fingerprint':fp,
            'frequencies':frequencies,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},path)
        path.replace(root/'last.pt')
    total_seconds=0.;new_tokens=0;total=c['training_tokens_per_predictor']
    for index in range(position,len(windows)):
        row=windows[index];ids=window_ids(sources,row);p.train()
        optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.monotonic()
        with torch.no_grad(),controller.use_gates(),capture_neuro(model) as trace:model.model(ids,use_cache=False)
        with torch.no_grad(): context,labels=target_ids(trace,thresholds,banks)
        features=feature_tuple(trace);del trace
        lr=c['learning_rate']*min(1.,(seen+ids.numel())/20000)*(.1+.45*(1+math.cos(math.pi*seen/total)))
        for group in optimizer.param_groups:group['lr']=lr
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=p(features,context)
            loss=sum(F.cross_entropy(scores.float().flatten(0,-2),target.flatten()) for scores,target in zip(logits,labels))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite centroid CE')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(p.parameters(),c['gradient_clip'],error_if_nonfinite=True)
        assert all(v.grad is None and not v.requires_grad for v in model.parameters())
        optimizer.step()
        with torch.no_grad():
            for branch,(scores,target) in enumerate(zip(logits,labels)):
                train_correct[branch]+=(scores.argmax(-1)==target).sum()
                for layer in range(26):frequencies[branch,layer]+=torch.bincount(target[...,layer].flatten(),minlength=c['centroids'])
            train_count+=labels[0].numel()
        torch.cuda.synchronize();total_seconds+=time.monotonic()-tick;new_tokens+=ids.numel()
        seen+=ids.numel();position=index+1
        milestone=position==len(windows) or (index+1<len(windows) and windows[index+1]['stage']!=row['stage'])
        checkpoint=seen-last_save>=c['checkpoint_every_tokens'] or milestone or stop
        if index%20==0 or checkpoint:
            record={'tokens':seen,'target':total,'tokens_per_second':new_tokens/max(total_seconds,1e-8),
                'stage':row['stage'],'loss':float(loss.detach()),'gradient_norm':float(norm),
                'training_stream_accuracy':(train_correct/max(train_count,1)).cpu().tolist(),
                'seconds':elapsed+time.monotonic()-started}
            atomic_json(root/'progress.json',record);print(json.dumps(record),flush=True)
        if checkpoint:save();last_save=seen
        if stop:atomic_json(root/'paused.json',{'tokens':seen});return 'paused'
        # Save/evaluate near requested milestones without altering planner windows.
        requested=next((m for m in c['checkpoint_tokens'] if seen>=m and not milestone_ready(root,m)),None)
        if requested is not None:
            results=accuracy(model,controller,p,banks,thresholds,corpus,validation,frequencies)
            save_milestone(root,p,requested,seen,fp,results)
            print(json.dumps({'stage':'validation_accuracy','tokens':seen,**results}),flush=True)
    assert seen==total
    # A process may resume from a final optimizer checkpoint written immediately
    # before evaluation; finish that evaluation without presenting more tokens.
    if not milestone_ready(root,total):
        final=accuracy(model,controller,p,banks,thresholds,corpus,validation,frequencies)
        save_milestone(root,p,total,seen,fp,final)
    final=json.loads((root/f'accuracy-{total}.json').read_text())
    if not (root/'runtime-overhead.json').exists():
        atomic_json(root/'runtime-overhead.json',runtime_overhead(model,controller,p,banks,thresholds,
            window_ids(sources,windows[0])))
    passed=min(final['gate_top1'],final['up_top1'])>=c['far_below_accuracy']
    status='complete' if passed else 'stopped_low_centroid_accuracy'
    atomic_json(root/'training.json',{'status':status,'tokens':seen,'seconds':elapsed+time.monotonic()-started,
        'accuracy':final,'threshold':c['far_below_accuracy'],'fingerprint':fp,
        'training_objective':'cross entropy; same13M planner token presentations, no output-quality fine-tuning'})
    return status


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/neuroprefetcher.json')
    parser.add_argument('--budget',required=True,type=float);a=parser.parse_args()
    c=json.loads(Path(a.config).read_text());assert a.budget in c['nominal_budgets']
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    status=train(c,a.budget)
    if status=='paused':raise SystemExit(75)
    if status=='stopped_low_centroid_accuracy':raise SystemExit(42)


if __name__=='__main__':main()
