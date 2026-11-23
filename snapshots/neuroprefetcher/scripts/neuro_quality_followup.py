"""User-authorized quality evaluation despite low centroid accuracy; no training."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json, file_sha256
from pathpred.neuro_calibration import banks_at_count, intersection_retention, calibrate_count
from scripts.neuro_train import load_banks, predictor, validation_data, identity
from scripts.neuro_evaluate import validation_predictions, evaluate
from scripts.neuron_guessing import backbone
from scripts.report_neuroprefetcher import paired_intervals


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/neuroprefetcher.json')
    parser.add_argument('--root',default='artifacts/neuro-quality-v1')
    args=parser.parse_args();c=json.loads(Path(args.config).read_text())
    root=Path(args.root);launch=root/'launch';launch.mkdir(parents=True,exist_ok=True)
    start=time.time()
    atomic_json(launch/'progress.json',{'pid':os.getpid(),'stage':'quality','status':'running','started_epoch':start})
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        source=Path(c['root'])/'budget-50'
        training=json.loads((source/'training.json').read_text())
        assert training['tokens']==13000000
        assert training['fingerprint']==identity(c,.5)
        meta=json.loads((source/'checkpoint-13000000.json').read_text())
        weights=source/meta['file']['file'];assert file_sha256(weights)==meta['file']['sha256']
        protocol={'user_override':'User explicitly requested perplexity after reviewing centroid accuracy.',
            'training_status_preserved':training['status'],'predictor_tokens':training['tokens'],
            'predictor_sha256':meta['file']['sha256'],'training_updates':0,
            'targets':[.5,.4,.5162],'native_masks_also_evaluated':True,
            'forty_percent_policy':'Same nominal50%-vocabulary checkpoint calibrated to40%; no separately trained40% predictor.',
            'planner_retrained':False,'planner_rerun':False}
        if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
        else:atomic_json(root/'protocol.json',protocol)
        p=predictor(c);p.load_state_dict(load_file(str(weights),device='cuda'));p.eval()
        banks,thresholds,aggregates=load_banks(c,.5)
        model,controller=backbone();corpus,validation,test=validation_data()
        codes=validation_predictions(model,controller,p,banks,thresholds,corpus,validation,root/'validation-predictions.safetensors')
        measurements={}
        def measure(count):
            if count not in measurements:
                measurements[count]=intersection_retention(codes['gate'],codes['up'],banks_at_count(aggregates,count))
                print(json.dumps({'stage':'calibration','count':count,'retention':measurements[count]}),flush=True)
            return measurements[count]
        results={};paired={};planner={}
        for target in (.5,.4,.5162,None):
            label='native' if target is None else f'{target:.4f}'
            if target is None:
                cal={'policy':'Original training centroid masks; no density adjustment','matched_budget':False,
                     'validation_retention':intersection_retention(codes['gate'],codes['up'],banks)}
                active=banks
            else:
                calpath=root/f'calibration-{label}.json'
                if calpath.exists():cal=json.loads(calpath.read_text())
                else:
                    cal=calibrate_count(measure,target);cal['validation_documents']=validation
                    atomic_json(calpath,cal)
                active=banks_at_count(aggregates,cal['count_per_centroid'])
            result=evaluate(model,controller,p,banks,thresholds,active,corpus,test,root/f'evaluation-{label}.json',cal)
            results[label]=result
            basebudget=40 if target==.4 else 50
            base=Path(c['planner_artifact_root'])/'autoregressive'/f'budget-{basebudget}'
            planner[str(basebudget)]=json.loads((base/'evaluation.json').read_text())
            if target is not None and cal['matched_budget']:
                paired[label]=paired_intervals(json.loads((root/f'evaluation-{label}-documents.json').read_text()),
                    json.loads((base/'evaluation-documents.json').read_text()))
                paired[label]['planner_validation_target']=basebudget/100
                paired[label]['same_validation_target']=target==basebudget/100
            atomic_json(root/'results.json',{'protocol':protocol,'results':results,'saved_planner':planner,'paired':paired})
            print(json.dumps({'stage':'evaluation_complete','label':label,'overall':result['overall']}),flush=True)
        atomic_json(launch/'completion.json',{'status':'complete','seconds':time.time()-start,'training_updates':0,
            'quality_evaluations':list(results),'accuracy_stop_overridden_by_user':True})
    except BaseException as error:
        atomic_json(launch/'completion.json',{'status':'failed','error':str(error),'seconds':time.time()-start})
        raise


if __name__=='__main__':main()
