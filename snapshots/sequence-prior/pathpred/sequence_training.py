"""Student-recorded two-pass approximation; exact causal statistics on that record."""
import torch
from .end_to_end import dense_teacher,quality_loss
from .planner_route import planner_route,previous_context
from .sequence_prior import SequenceTracker,TrackerConfig,relative_energy,causal_scores


def stacked(trace,key,layers):return torch.stack([trace[key][l] for l in range(layers)],2)


@torch.no_grad()
def prompt_teacher(model,controller,ids,prompt_tokens):
    """Only legitimate dense-prompt activations may seed the recording policy."""
    energy={};handles=[]
    for layer,module in enumerate(model.model.layers):
        def hook(module,args,l=layer):
            a=args[0][:,:prompt_tokens].detach()
            energy[l]=relative_energy(a,torch.ones_like(a,dtype=torch.bool))
        handles.append(module.mlp.down_proj.register_forward_pre_hook(hook))
    try:teacher,pre=dense_teacher(model,controller,ids)
    finally:
        for h in handles:h.remove()
    return teacher,pre,torch.stack([energy[l] for l in range(len(model.model.layers))],2)


def end_to_end_batch(model,controller,planner,optimizer,thresholds,ids,offsets,dual,*,prompt_tokens=64,
                     tracker_config=TrackerConfig(),mode='gumbel',temperature=1.,beta=1.,epsilon=.01,chunk_size=32):
    b,t=ids.shape;c=planner.config
    if not 0<prompt_tokens<t:raise ValueError('Prompt must precede training suffix')
    optimizer.zero_grad(set_to_none=True);planner.eval()
    dense_tokens=(torch.arange(t,device=ids.device)[None,:]<prompt_tokens).expand(b,-1)
    teacher,teacher_pre,prompt_energy=prompt_teacher(model,controller,ids,prompt_tokens)
    with torch.no_grad():
        tracker=SequenceTracker(b,c.llm_layers,c.units,device=ids.device,config=tracker_config)
        tracker.update(prompt_energy,torch.ones_like(prompt_energy,dtype=torch.bool),prompt=True)
        # Bootstrap pass: prompt-only prior and unavailable previous-path context.
        # It is NOT a fixed-point or exact sequential on-policy rollout.
        seed=tracker.scores()[:,None].expand(-1,t,-1,-1)
        with planner_route(model,controller,planner,thresholds,mode='hard',offsets=offsets,keep=dual.target,
                checkpoint_layers=False,sequence_scores=seed,dense_tokens=dense_tokens,record_energy=True) as recorded:
            model.model(ids,use_cache=False)
        bits=stacked(recorded,'bits',c.llm_layers)
        previous,valid=previous_context(bits)
        scores=causal_scores(stacked(recorded,'energy',c.llm_layers),stacked(recorded,'observed',c.llm_layers),
                             prompt_tokens,tracker_config)
        del recorded,bits,tracker,seed,prompt_energy
    planner.train()
    with planner_route(model,controller,planner,thresholds,previous=previous,previous_valid=valid,
            mode=mode,temperature=temperature,keep=dual.target,offsets=offsets,
            sequence_scores=scores,dense_tokens=dense_tokens) as trace:
        hidden=model.model(ids,use_cache=False).last_hidden_state
        loss,metrics=quality_loss(model,hidden,trace['pre_norm'],teacher,teacher_pre,trace,
            alpha=dual.alpha,beta=beta,epsilon=epsilon,chunk_size=chunk_size,token_mask=~dense_tokens)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sequence-prior loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(planner.parameters(),1.,error_if_nonfinite=True)
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        optimizer.step();result={k:float(v.detach()) for k,v in metrics.items()}
        result.update(loss=float(loss.detach()),gradient_norm=float(norm),alpha=dual.alpha,
            mean_prior_gain=float(planner.prior_gain.mean().detach()),
            max_abs_prior_gain=float(planner.prior_gain.detach().abs().max()),
            whole_plan_calls=trace['predictor_calls'],recomputations=trace['recomputations'],
            history='Prompt-seeded bootstrap two-pass; exact prefix sums on student observations, approximate policy history')
    dual.update(result['used']);return result
