"""Single-sequence decode latency and explicit byte-payload accounting.

No hardware DRAM counter is claimed. The prototype still executes dense Qwen
matmuls; these timings measure routing overhead, not a sparse-kernel speedup.
"""
import json
from pathlib import Path
import statistics
import time
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json
from pathpred.neuron_prediction import NeuronPredictor,neuron_config,route_neurons
from scripts.neuron_guessing import ROOT,CAL,backbone,training_order


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    training=json.loads((ROOT/'training.json').read_text())
    records=[json.loads(line) for line in Path('data/neuron-training-corpus-v1/sequences.jsonl').read_text().splitlines()]
    used=[records[i] for i in training_order(records)[:training['distinct_windows']]]
    heldout=[json.loads(line) for line in Path('data/stage3-corpus-v1/sequences.jsonl').read_text().splitlines()]
    forbidden={r['document_sha256'] for r in heldout if r['split']!='train'}
    assert all(r['split']=='train' and r['document_sha256'] not in forbidden for r in used)
    atomic_json(ROOT/'training-data-audit.json',{'windows':len(used),'distinct_documents':len({r['document_sha256'] for r in used}),
        'no_repeated_windows':len({r['sequence_id'] for r in used})==len(used),
        'disjoint_from_validation_and_test':True,'tokens_per_predictor':training['tokens_per_predictor']})
    model,controller=backbone();cal=load_file(str(CAL),device='cuda')
    corpus=load_file('data/stage3-corpus-v1/tokens.safetensors')['input_ids']
    prompt=corpus[:1,:512].long().cuda();continuation=corpus[1:2,:25].long().cuda()
    result={'gpu':torch.cuda.get_device_name(),'context_start':512,'timed_tokens':20,
            'warmup_tokens':5,'measurements':{},'notes':[
                'Qwen uses the canonical grouped-checkpoint reference with weight gathers and dense matmuls in every method; this is not an optimized production dense baseline or sparse speedup.',
                'Wall-clock timings include Python and CUDA synchronization.',
                'Byte figures are estimated weight payloads, not measured DRAM traffic.',
                'Nsight Compute is not installed; cache effects and runtime traffic remain unmeasured.',
                'BF16 matrix storage preserves FP32 layer embeddings, output bias, normalization and residual terms; cached logits are checked against original FP32 parameters under BF16 autocast.']}
    reference_outputs={}
    for storage in ['fp32','bf16_matrices']:
        for signal in ['dense','binary','hidden']:
            if signal=='dense' and storage=='bf16_matrices':continue
            p=None
            if signal!='dense':
                p=NeuronPredictor(neuron_config(signal)).cuda()
                p.load_state_dict(load_file(str(ROOT/f'{signal}.safetensors'),device='cuda'));p.eval()
                if storage=='bf16_matrices':
                    for name,parameter in p.named_parameters():
                        if parameter.ndim>=2 and name not in ('layer_embedding','output_bias'):
                            parameter.data=parameter.data.to(torch.bfloat16)
            for keep in ([1.] if p is None else [.5,.4]):
                previous={}
                with controller.use_gates():
                    if p is None:
                        cache=model.model(prompt,use_cache=True).past_key_values
                    else:
                        with route_neurons(model,controller,cal['q75'],keep,predictor=p,next_embeddings=previous):
                            cache=model.model(prompt,use_cache=True).past_key_values
                durations=[];row_counts=[]
                for t in range(25):
                    next_embeddings={};torch.cuda.synchronize();tick=time.monotonic()
                    with controller.use_gates():
                        if p is None:
                            out=model(continuation[:,t:t+1],past_key_values=cache,use_cache=True)
                        else:
                            with route_neurons(model,controller,cal['q75'],keep,predictor=p,
                                previous_embeddings=previous,next_embeddings=next_embeddings) as trace:
                                out=model(continuation[:,t:t+1],past_key_values=cache,use_cache=True)
                    torch.cuda.synchronize();seconds=time.monotonic()-tick
                    cache=out.past_key_values;previous=next_embeddings
                    if t>=5:
                        durations.append(seconds)
                        if signal=='binary':
                            rows=sum(int((trace['bits'][l]>=0).sum())+int((trace['bits'][l]==1).sum()) for l in range(26))
                            row_counts.append(rows)
                    if p is not None:
                        values=out.logits.detach().cpu()
                        if storage=='fp32':reference_outputs[signal,keep,t]=values
                        else:assert torch.equal(values,reference_outputs[signal,keep,t]), (signal,keep,t)
                    del out
                key=f'{signal}-{round(keep*100)}-{storage}'
                item={'mean_ms':statistics.mean(durations)*1000,'median_ms':statistics.median(durations)*1000,
                    'tokens_per_second':1/statistics.mean(durations),'seconds':durations}
                if p is not None:
                    item['actual_parameter_storage_bytes']=sum(v.numel()*v.element_size() for v in p.parameters())
                    item['cached_logits_match_fp32_autocast']=True if storage=='bf16_matrices' else None
                if row_counts:
                    item['ideal_gather_input_bytes_bf16']=statistics.mean(row_counts)*256*2
                    item['output_projection_bytes_bf16']=26*256*6144*2+26*6144*2
                    item['full_input_projection_bytes_bf16']=26*2*6144*256*2
                    unique_parameters=sum(v.numel() for v in p.parameters())
                    distinct_parameters=sum(v.numel() for v in [p.activity_projection,p.output_weight,p.output_bias,p.layer_embedding])
                    shared_parameters=unique_parameters-distinct_parameters
                    item['dense_weight_payload_cached_shared_bf16']=unique_parameters*2
                    item['dense_weight_payload_reread_shared_bf16']=(distinct_parameters+26*shared_parameters)*2
                    item['gather_weight_payload_cached_shared_bf16']=unique_parameters*2-item['full_input_projection_bytes_bf16']+item['ideal_gather_input_bytes_bf16']
                    item['gather_weight_payload_reread_shared_bf16']=item['gather_weight_payload_cached_shared_bf16']+25*shared_parameters*2
                    item['ideal_qwen_mlp_weight_bytes_omitted']=(1-keep)*26*3*2048*6144*2
                result['measurements'][key]=item
                del cache
            del p
    atomic_json(ROOT/'decode-profile.json',result)
    print(json.dumps(result),flush=True)
    from scripts.verify_neuron_decode import verify
    verify(model,controller)


if __name__=='__main__':main()
