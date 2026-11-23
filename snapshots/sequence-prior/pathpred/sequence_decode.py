"""Exact token-by-token deployment of the tracker and whole-path planner."""
import torch
from torch.nn import functional as F
from .planner_route import planner_route
from .sequence_prior import SequenceTracker,TrackerConfig,relative_energy
from .sequence_training import stacked


@torch.no_grad()
def decode_batch(model,controller,planner,thresholds,ids,lengths,*,offsets,keep,prompt_tokens=64,
                 tracker_config=TrackerConfig(),quality=True,progress=None):
    b,t=ids.shape;c=planner.config;device=ids.device
    if lengths.shape!=(b,) or (lengths<=prompt_tokens+1).any():raise ValueError('Documents must have a scored continuation')
    attention=torch.arange(t,device=device)[None,:]<lengths[:,None]
    tracker=SequenceTracker(b,c.llm_layers,c.units,device=device,config=tracker_config)
    bits={};handles=[]
    # Dense prefill: all neurons observed, only the final prompt path retained.
    for layer,module in enumerate(model.model.layers):
        def hook(module,args,l=layer):
            a=args[0].detach();e=relative_energy(a,torch.ones_like(a,dtype=torch.bool))
            tracker.total[:,l].add_(e.sum(1)*tracker_config.prompt_weight)
            tracker.count[:,l].add_(a.shape[1]*tracker_config.prompt_weight)
            bits[l]=(a[:,-1:].float().abs()>thresholds[l]).to(torch.int8)
        handles.append(module.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with controller.use_gates():
            out=model.model(ids[:,:prompt_tokens],attention_mask=attention[:,:prompt_tokens],use_cache=True)
    finally:
        for h in handles:h.remove()
    student_cache=out.past_key_values;previous=torch.stack([bits[l] for l in range(c.llm_layers)],2)
    teacher_cache=None
    if quality:
        with controller.use_gates():
            teacher_cache=model.model(ids[:,:prompt_tokens],attention_mask=attention[:,:prompt_tokens],use_cache=True).past_key_values
    sums=torch.zeros(b,5,device=device,dtype=torch.float64);kept_sum=sums[:,0].clone();kept_square=kept_sum.clone()
    ever=torch.zeros(b,planner.steps,c.units,device=device,dtype=torch.bool)
    reactivated=kept_sum.clone();eligible=kept_sum.clone();old_selected=None
    per_layer=torch.zeros(b,planner.steps,device=device,dtype=torch.float64)
    for token in range(prompt_tokens,int(lengths.max())-1):
        if progress is not None and token%64==0:progress(token,int(lengths.max())-1)
        current=ids[:,token:token+1];mask=attention[:,:token+1];valid=token+1<lengths
        # Snapshot BEFORE executing any current-token layer; update only after Qwen finishes.
        prior=tracker.scores()[:,None]
        with planner_route(model,controller,planner,thresholds,previous=previous,
                previous_valid=torch.ones(b,1,device=device,dtype=torch.bool),sequence_scores=prior,
                offsets=offsets,mode='hard',keep=keep,checkpoint_layers=False,record_energy=True) as trace:
            student=(model if quality else model.model)(current,attention_mask=mask,past_key_values=student_cache,use_cache=True)
        student_cache=student.past_key_values
        previous=stacked(trace,'bits',c.llm_layers)
        tracker.update(stacked(trace,'energy',c.llm_layers),stacked(trace,'observed',c.llm_layers),valid=valid[:,None])
        selected=torch.stack([trace['gates'][l][:,0]>=.5 for l in range(c.dense_prefix,c.llm_layers)],1)
        kept=selected.float().mean((1,2));kept_sum+=kept*valid;kept_square+=kept.double().square()*valid
        per_layer+=selected.float().mean(-1)*valid[:,None];ever|=selected & valid[:,None,None]
        if old_selected is not None:
            eligible+=(~old_selected).sum((1,2))*valid
            reactivated+=(selected & ~old_selected).sum((1,2))*valid
        old_selected=selected;sums[:,4]+=valid
        if quality:
            with controller.use_gates():teacher=model(current,attention_mask=mask,past_key_values=teacher_cache,use_cache=True)
            teacher_cache=teacher.past_key_values
            a=teacher.logits[:,0].float();z=student.logits[:,0].float();al=F.log_softmax(a,-1);zl=F.log_softmax(z,-1)
            target=ids[:,token+1,None]
            values=torch.stack((-al.gather(-1,target)[:,0],-zl.gather(-1,target)[:,0],
                (al.exp()*(al-zl)).sum(-1),(a.argmax(-1)==z.argmax(-1)).float()),1)
            sums[:,:4]+=values.double()*valid[:,None]
    return {k:v.cpu() for k,v in dict(sums=sums,kept_sum=kept_sum,kept_square=kept_square,
        per_layer=per_layer,reactivated=reactivated,reactivation_opportunities=eligible,
        never_selected_fraction=(~ever).float().mean((1,2))).items()}
