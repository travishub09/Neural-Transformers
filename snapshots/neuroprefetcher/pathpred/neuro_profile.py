"""Measured batch1 predictor overhead; does not claim sparse speedup."""
import statistics
import time
import torch
from .neuro_features import capture_neuro,feature_tuple,predicted_masks
from .neuroprefetcher import nearest_centroid


@torch.no_grad()
def runtime_overhead(model,controller,p,banks,thresholds,seed_ids,context=1024,repetitions=20):
    p.eval();device=next(p.parameters()).device
    # Synthetic context, used only for timing; never contributes quality metrics.
    ids=seed_ids[:1].repeat(1,(context+seed_ids.shape[1]-1)//seed_ids.shape[1])[:,:context]
    with controller.use_gates():out=model(ids,use_cache=True)
    cache=out.past_key_values;token=ids[:,-1:];dense=[]
    for iteration in range(repetitions+3):
        torch.cuda.synchronize();start=time.perf_counter()
        with controller.use_gates():out=model(token,past_key_values=cache,use_cache=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;cache=out.past_key_values
        if iteration>=3:dense.append(elapsed*1000)
    with controller.use_gates(),capture_neuro(model,targets=False) as trace:
        model(token,past_key_values=cache,use_cache=True)
    features=feature_tuple(trace);key=f'gate_{p.config.prefix-1}';times=[]
    for iteration in range(repetitions+3):
        torch.cuda.synchronize();start=time.perf_counter()
        context_id=nearest_centroid(trace['activations'][(p.config.prefix-1,'gate')].abs()>=thresholds[key],banks[key])
        with torch.autocast('cuda',dtype=torch.bfloat16):gate,up=p(features,context_id)
        masks=predicted_masks(gate.argmax(-1),up.argmax(-1),banks)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if iteration>=3:times.append(elapsed*1000)
    a=statistics.median(times);b=statistics.median(dense)
    return {'predictor_matching_lookup_ms_median':a,'dense_qwen_decode_ms_median':b,
        'predictor_to_dense_decode_latency_ratio':a/b,'predictor_samples_ms':times,
        'dense_samples_ms':dense,'batch_size':1,'initial_context':context,
        'dense_measured_context_range':[context+3,context+repetitions+2],
        'repetitions':repetitions,'synchronization':'CUDA synchronize around wall-clock timing',
        'includes':['prefix centroid nearest matching','full classifier','argmax and centroid mask intersection',
                    'actual fp32-parameter/bf16-autocast conversion and launch overhead'],
        'excludes':['prefix feature capture/sketch already obtained from Qwen','sparse decoder kernels','storage prefetching'],
        'quality_data_reused_for_timing_only':True,'sparse_speedup_claim':False,
        'gpu_name':torch.cuda.get_device_name(device),'torch_version':torch.__version__}
