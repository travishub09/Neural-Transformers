"""Exact causal previous-token histories, batched over independent documents.

Language positions execute sequentially; batching only combines different
documents. Both dense teacher and sparse student use their own real KV caches.
This execution protocol is reported separately from old full-sequence scores.
"""
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from .planner_route import planner_route
from .neuron_prediction import collect_dense,route_neurons


@torch.no_grad()
def decode_batch(model,controller,predictor,thresholds,ids,lengths,*,offsets,keep,
                 closed_loop=False,drop_previous=False,quality=True,progress=None):
    if ids.ndim!=2 or lengths.shape!=(ids.shape[0],):raise ValueError('Invalid batch')
    if ((lengths<2)|(lengths>ids.shape[1])).any():raise ValueError('Invalid document lengths')
    device=ids.device;b=ids.shape[0];layers=len(model.model.layers);units=thresholds.shape[1]
    teacher_cache=student_cache=None;previous=None;previous_embeddings=None
    maximum=int(lengths.max())
    attention=torch.arange(ids.shape[1],device=device)[None,:]<lengths[:,None]
    sums=torch.zeros(b,5,device=device,dtype=torch.float64)
    # Columns: true positives, teacher positives, selected, union (per layer).
    layer_counts=torch.zeros(b,layers-2,4,device=device,dtype=torch.float64)
    kept_sum=torch.zeros(b,device=device,dtype=torch.float64)
    prefix=2 if closed_loop else predictor.config.dense_prefix
    if prefix!=2:raise ValueError('This comparison fixes two dense layers')
    for token in range(maximum-1):
        if progress is not None and token % 32 == 0:
            progress(token, maximum-1)
        current=ids[:,token:token+1]
        mask=attention[:,:token+1]
        valid=token+1<lengths
        if quality:
            with controller.use_gates(),collect_dense(model,thresholds,budgets=(keep,),capture_hidden=False) as teacher_trace:
                teacher_out=model(current,attention_mask=mask,past_key_values=teacher_cache,use_cache=True)
            teacher_cache=teacher_out.past_key_values
        if closed_loop:
            next_embeddings={}
            with controller.use_gates(),route_neurons(model,controller,thresholds,keep,predictor=predictor,
                    previous_embeddings=previous_embeddings,next_embeddings=next_embeddings,
                    score_offsets=offsets) as trace:
                student_out=(model if quality else model.model)(current,attention_mask=mask,
                    past_key_values=student_cache,use_cache=True)
            previous_embeddings=next_embeddings;gates=trace['masks']
        else:
            history=previous if predictor.config.previous_path else None
            history_valid=torch.full((b,1),token>0,device=device,dtype=torch.bool) if history is not None else None
            with planner_route(model,controller,predictor,thresholds,previous=history,
                    previous_valid=history_valid,offsets=offsets,mode='hard',keep=keep,
                    checkpoint_layers=False,drop_previous=drop_previous) as trace:
                student_out=(model if quality else model.model)(current,attention_mask=mask,
                    past_key_values=student_cache,use_cache=True)
            previous=torch.stack([trace['bits'][l] for l in range(layers)],dim=2)
            gates=trace['gates']
        student_cache=student_out.past_key_values
        selected=torch.stack([gates[l][:,0]>=.5 for l in range(prefix,layers)],dim=1)
        kept_sum+=(selected.double().mean((1,2))*valid)
        sums[:,4]+=valid
        if quality:
            a=teacher_out.logits[:,0].float();z=student_out.logits[:,0].float()
            teacher_log=F.log_softmax(a,dim=-1);student_log=F.log_softmax(z,dim=-1)
            target=ids[:,token+1,None]
            values=torch.stack((-teacher_log.gather(-1,target)[:,0],-student_log.gather(-1,target)[:,0],
                (teacher_log.exp()*(teacher_log-student_log)).sum(-1),
                (a.argmax(-1)==z.argmax(-1)).float()),dim=1)
            sums[:,:4]+=values.double()*valid[:,None]
            oracle=torch.stack([teacher_trace['labels'][l][str(keep)][:,0] for l in range(prefix,layers)],dim=1)
            counts=torch.stack(((selected&oracle).sum(-1),oracle.sum(-1),selected.sum(-1),
                                (selected|oracle).sum(-1)),dim=-1)
            layer_counts+=counts.double()*valid[:,None,None]
    return {'sums':sums.cpu(),'kept_sum':kept_sum.cpu(),'layer_counts':layer_counts.cpu(),
        'execution':'Sequential language positions with actual previous paths; independent documents batched',
        'dense_teacher_execution':'Sequential cached teacher in the same batch' if quality else None}
