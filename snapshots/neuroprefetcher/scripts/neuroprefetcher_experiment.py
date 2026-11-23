"""Real-GPU correctness/throughput preflight; never trains the upfront planner."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from pathpred.collection import atomic_json
from pathpred.neuro_features import capture_neuro,feature_tuple,target_ids,predicted_masks
from pathpred.neuroprefetcher import qwen_overhead,fit_centroids
from scripts.neuron_guessing import backbone
from scripts.neuro_data import load_matching,window_ids
from scripts.neuro_train import predictor
from scripts.neuro_train import validation_data


def smoke(c):
    root=Path(c['root']);root.mkdir(parents=True,exist_ok=True)
    dependency=Path(c['dependency'])
    assert dependency.exists(), 'Layer-only pilot must finish before this GPU task'
    assert json.loads(dependency.read_text())['status']=='complete', 'Resolve incomplete layer-only pilot first'
    sources,windows,audit=load_matching(c);atomic_json(root/'data-audit.json',audit)
    _,validation,test=validation_data()
    baseline=Path(c['planner_artifact_root'])/'autoregressive/budget-50'
    saved_validation=json.loads((baseline/'threshold.json').read_text())['identity']['validation_documents']
    saved_test=json.loads((baseline/'evaluation-documents.json').read_text())
    assert validation==saved_validation,'Validation documents differ from saved planner'
    assert len(test)==63
    assert {r['document_sha256'] for r in test}=={r['document'] for r in saved_test}
    assert {r['document_sha256']:min(r['length'],512)-1 for r in test}=={r['document']:r['tokens'] for r in saved_test}
    atomic_json(root/'comparison-inputs.json',{'validation':validation,'test':test,
        'saved_planner_documents_file':str(baseline/'evaluation-documents.json'),
        'source':'Read-only comparison with existing planner artifacts; no rerun'})
    model,controller=backbone();ids=window_ids(sources,windows[0]);p=predictor(c)
    opt=torch.optim.AdamW(p.parameters(),lr=c['learning_rate']);speeds=[]
    with torch.no_grad(),controller.use_gates(),capture_neuro(model) as trace:
        dense=model(ids,use_cache=False).logits
    with torch.no_grad(),controller.use_gates({(l,'mlp'):torch.ones(6144,device='cuda') for l in range(2,28)}):
        all_open=model(ids,use_cache=False).logits
    assert torch.equal(dense,all_open)
    torch.manual_seed(c['seed']);banks={};thresholds={}
    small_fit=fit_centroids(torch.rand(512,64,device='cuda')>.5,k=8,iterations=2,seed=c['seed'])
    assert small_fit['masks'].sum(-1).eq(32).all() and int(small_fit['counts'].sum())==512
    for (layer,projection),value in trace['activations'].items():
        key=f'{projection}_{layer}'
        banks[key]=torch.rand(c['centroids'],model.config.intermediate_size,device='cuda')>.5
        thresholds[key]=value.float().abs().flatten().median()
    for _ in range(2):
        torch.cuda.synchronize();started=time.monotonic();opt.zero_grad(set_to_none=True)
        with torch.no_grad(),controller.use_gates(),capture_neuro(model) as trace:model.model(ids,use_cache=False)
        with torch.no_grad():context,labels=target_ids(trace,thresholds,banks)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            gate,up=p(feature_tuple(trace),context)
            loss=F.cross_entropy(gate.float().flatten(0,-2),labels[0].flatten())+F.cross_entropy(up.float().flatten(0,-2),labels[1].flatten())
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert all(v.grad is None and not v.requires_grad for v in model.parameters())
        torch.cuda.synchronize();speeds.append(ids.numel()/(time.monotonic()-started))
    p.eval()
    with torch.no_grad():
        with torch.autocast('cuda',dtype=torch.bfloat16):gate,up=p(feature_tuple(trace),context)
        masks=predicted_masks(gate.argmax(-1),up.argmax(-1),banks)
        with controller.use_gates({(layer,'mlp'):value for layer,value in masks.items()}),capture_neuro(model) as student_trace:
            sparse=model(ids,use_cache=False).logits
        assert torch.isfinite(sparse).all() and not torch.equal(sparse,dense)
        for original,student in zip(feature_tuple(trace),feature_tuple(student_trace)):
            torch.testing.assert_close(original,student,atol=0,rtol=0)
    result={'all_open_exact':True,'frozen_qwen':True,'finite_gradient_norm':float(norm),'tokens_per_second':speeds,
        'throughput_excludes_centroid_target_matching':False,'parameters':qwen_overhead(p,model),
        'dense_prefix_features_unchanged_by_suffix_masks':True,'real_model_centroid_label_and_mask_pipeline':True,
        'cuda_centroid_fit_smoke':True,
        'centroids_for_smoke':'Synthetic random banks; discarded; no scientific quality result',
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'smoke_updates_discarded':True}
    atomic_json(root/'gpu-smoke.json',result);print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/neuroprefetcher.json');parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();assert a.smoke;c=json.loads(Path(a.config).read_text())
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');smoke(c)


if __name__=='__main__':main()
