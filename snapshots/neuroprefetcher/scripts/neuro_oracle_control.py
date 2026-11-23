"""Remove centroid-classification error without training or touching test labels in calibration."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors
from pathpred.neuro_features import capture_neuro, target_ids
from pathpred.neuroprefetcher import nearest_centroid
from pathpred.neuro_calibration import banks_at_count, intersection_retention, calibrate_count
from scripts.neuro_train import load_banks, validation_data
from scripts.neuro_evaluate import evaluate
from scripts.neuron_guessing import backbone
from scripts.planner_experiment import batches


@torch.no_grad()
def oracle_patterns(model,controller,banks,thresholds,corpus,validation,path):
    if path.exists():return load_file(str(path),device='cuda')
    patterns={f'{branch}_{layer}':[] for layer in range(2,28) for branch in ('gate','up')}
    for _,ids,lengths in batches(corpus,validation,16):
        cache=None;attention=torch.arange(ids.shape[1],device='cuda')[None,:]<lengths[:,None]
        for token in range(int(lengths.max())-1):
            valid=token+1<lengths
            with controller.use_gates(),capture_neuro(model,targets=True) as trace:
                out=model.model(ids[:,token:token+1],attention_mask=attention[:,:token+1],past_key_values=cache,use_cache=True)
            cache=out.past_key_values
            for layer in range(2,28):
                for branch in ('gate','up'):
                    key=f'{branch}_{layer}'
                    bits=trace['activations'][layer,branch][:,0].abs()>=thresholds[key]
                    patterns[key].append(bits[valid].cpu())
            if token%64==0:print(json.dumps({'stage':'oracle_validation','position':token}),flush=True)
    values={k:torch.cat(v) for k,v in patterns.items()};atomic_tensors(path,values)
    return {k:v.cuda() for k,v in values.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/neuro-oracle-v1');a=p.parse_args()
    root=Path(a.root);launch=root/'launch';launch.mkdir(parents=True,exist_ok=True);start=time.time()
    atomic_json(launch/'progress.json',{'pid':os.getpid(),'stage':'oracle','status':'running','started_epoch':start})
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        c=json.loads(Path('configs/neuroprefetcher.json').read_text())
        banks,thresholds,aggregates=load_banks(c,.5);model,controller=backbone()
        corpus,validation,test=validation_data()
        atomic_json(root/'protocol.json',{'selection':'Exact Hamming-nearest centroid of dense-teacher branch masks',
            'predictor':None,'training_updates':0,'target':.5,'dense_prefix':2,
            'caveat':'Privileged label control, not globally output-optimal centroid selection',
            'calibration':'Validation only; expand/prune aggregate masks and recompute exact nearest IDs for every candidate vocabulary',
            'down':'Full dense output projection on masked intermediate',
            'validation_documents':validation,'evaluation_documents':test})
        patterns=oracle_patterns(model,controller,banks,thresholds,corpus,validation,root/'validation-patterns.safetensors')
        def measure(count):
            active=banks_at_count(aggregates,count)
            codes={branch:torch.stack([nearest_centroid(patterns[f'{branch}_{layer}'],active[f'{branch}_{layer}'])
                for layer in range(2,28)],-1) for branch in ('gate','up')}
            used=intersection_retention(codes['gate'],codes['up'],active)
            print(json.dumps({'stage':'calibration','count':count,'retention':used}),flush=True);return used
        cal=calibrate_count(measure,.5);atomic_json(root/'calibration.json',cal)
        assert cal['matched_budget'],cal
        active=banks_at_count(aggregates,cal['count_per_centroid'])
        result=evaluate(model,controller,None,banks,thresholds,active,corpus,test,root/'evaluation.json',cal,oracle=True)
        assert result['overall']['gate_centroid_top1']==1 and result['overall']['up_centroid_top1']==1
        atomic_json(launch/'completion.json',{'status':'complete','seconds':time.time()-start,'overall':result['overall']})
    except BaseException as e:
        atomic_json(launch/'completion.json',{'status':'failed','error':str(e),'seconds':time.time()-start});raise


if __name__=='__main__':main()
