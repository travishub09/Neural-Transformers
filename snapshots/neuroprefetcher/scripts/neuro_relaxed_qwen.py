"""Inference-only larger dense-prefix comparison of saved Qwen predictors."""
import argparse
import functools
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json,file_sha256
from pathpred.neuro_calibration import banks_at_count,intersection_retention,calibrate_count
from scripts.neuro_train import load_banks,predictor,validation_data
from scripts.neuro_evaluate import evaluate,validation_predictions
from scripts.neuron_guessing import backbone,CAL
from scripts import planner_experiment as pe
from scripts.report_neuroprefetcher import paired_intervals


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    c=json.loads(Path('configs/neuroprefetcher.json').read_text());pc=json.loads(Path(c['planner_config']).read_text())
    prefix=20;banks,thresholds,aggregates=load_banks(c,.5)
    model,controller=backbone();corpus,validation,test=validation_data()
    p=predictor(c);source=Path(c['root'])/'budget-50'
    meta=json.loads((source/'checkpoint-13000000.json').read_text());weights=source/meta['file']['file']
    assert file_sha256(weights)==meta['file']['sha256'];p.load_state_dict(load_file(str(weights),device='cuda'));p.eval()
    # Dense-prefix features are unchanged; later forced-dense layers do not leak into predictions.
    codes=validation_predictions(model,controller,p,banks,thresholds,corpus,validation,root/'validation-codes.safetensors')
    def measure(count):
        return intersection_retention(codes['gate'][:,prefix-2:],codes['up'][:,prefix-2:],banks_at_count(aggregates,count),prefix=prefix)
    cal=calibrate_count(measure,.5);atomic_json(root/'neuro-calibration.json',cal);assert cal['matched_budget']
    nr=evaluate(model,controller,p,banks,thresholds,banks_at_count(aggregates,cal['count_per_centroid']),
        corpus,test,root/'neuro-evaluation.json',cal,dense_prefix=prefix)
    del p
    p=pe.planner('autoregressive',pc);base=Path(c['planner_artifact_root'])/'autoregressive'/'budget-50'
    pe.load_stage_weights(p,base,2000000);p.eval()
    q=load_file(str(CAL),device='cuda')['q75']
    offsets=torch.tensor(json.loads((base/'threshold.json').read_text())['offsets'],device='cuda')
    pe.decode_batch=functools.partial(pe.decode_batch,execution_dense_prefix=prefix)
    calibrated=pe.calibrate(model,controller,p,q,offsets,corpus,validation,.5,root/'planner-calibration.json',pc)
    pr=pe.evaluate(model,controller,p,q,calibrated,corpus,test,.5,root/'planner-evaluation.json',pc)
    for row in pr['per_layer_oracle_accuracy']:row['layer_one_based']+=prefix-2
    pr['execution_dense_prefix']=prefix;atomic_json(root/'planner-evaluation.json',pr)
    pair=paired_intervals(json.loads((root/'neuro-evaluation-documents.json').read_text()),json.loads((root/'planner-evaluation-documents.json').read_text()))
    atomic_json(root/'results.json',{'neuro':nr,'planner':pr,'paired':pair,
        'protocol':{'training_updates':0,'execution_dense_prefix':prefix,'planning_context_prefix':2,
            'budget':'50% of neurons in layers21–28, calibrated separately on validation',
            'down':'Full dense projection of masked MLP intermediate for both',
            'limitation':'Qwen execution-prefix control; saved predictors retain their original two-layer context. Not a Mistral-trained planner.'}})


if __name__=='__main__':main()
