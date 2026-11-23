"""Execute a complete precomputed suffix plan and record observed activity.

The parallel two-pass helper is intentionally described as an approximation:
its recording pass starts without previous-path context. Exact causal decoding
supplies the actual preceding token's observations from its completed rollout.
"""
from contextlib import contextmanager,nullcontext
import torch
from torch.utils.checkpoint import checkpoint


def previous_context(observations):
    if observations.ndim!=4:raise ValueError('Expected [batch,tokens,layers,units]')
    previous=torch.cat((torch.full_like(observations[:,:1],-1),observations[:,:-1]),dim=1)
    valid=torch.arange(observations.shape[1],device=observations.device)[None,:]>0
    return previous,valid.expand(observations.shape[0],-1)


@contextmanager
def planner_route(model,controller,planner,thresholds,*,previous=None,previous_valid=None,
                  mode='hard',temperature=1.,keep=.5,offsets=None,checkpoint_layers=True,
                  drop_previous=False,teacher_masks=None,teacher_probability=0.,execution_dense_prefix=None):
    if controller.state.active:raise RuntimeError('planner_route owns its gate context')
    c=planner.config
    dense_until=c.dense_prefix if execution_dense_prefix is None else execution_dense_prefix
    if not c.dense_prefix<=dense_until<c.llm_layers:raise ValueError('Invalid execution prefix')
    if len(model.model.layers)!=c.llm_layers:raise ValueError('Planner/backbone layer mismatch')
    if thresholds.shape!=(c.llm_layers,c.units):raise ValueError('Threshold shape mismatch')
    trace={'bits':{},'gates':{},'soft':{},'scores':{},'pre_norm':None,
           'predictor_calls':0,'plan':None,'recomputations':0,'events':[]}
    handles=[];originals=[];recomputing=False

    @contextmanager
    def recompute_context():
        nonlocal recomputing
        before=recomputing;recomputing=True;trace['recomputations']+=1
        try:yield
        finally:recomputing=before

    def checkpoint_forward(layer_id,original):
        def wrapped(hidden,*args,**kwargs):
            gate=controller.state.gates.get((layer_id,'mlp'))
            if gate is None or not torch.is_grad_enabled():return original(hidden,*args,**kwargs)
            def run(h,g):
                if not controller.state.active:raise RuntimeError('Backward must remain inside planner_route')
                controller.state.gates[layer_id,'mlp']=g
                return original(h,*args,**kwargs)
            return checkpoint(run,hidden,gate,use_reentrant=False,
                              context_fn=lambda:(nullcontext(),recompute_context()))
        return wrapped

    def observe(layer_id):
        def hook(module,args):
            if recomputing:return
            a=args[0].detach().float().abs()
            if layer_id<dense_until:
                gate=torch.ones_like(a,dtype=torch.bool);trace['gates'][layer_id]=gate
            else:
                # This assertion enforces the prefetch contract, independently
                # of whether every eventual mask happens to be all-on.
                if trace['plan'] is None or len(trace['scores'])!=planner.steps:
                    raise RuntimeError('Suffix execution began before a complete plan existed')
                gate=trace['gates'][layer_id].detach()>=.5
            trace['events'].append(('execute',layer_id))
            trace['bits'][layer_id]=torch.where(gate,(a>thresholds[layer_id]).to(torch.int8),-1)
        return hook

    def make_plan(module,args,output):
        if trace['plan'] is not None:raise RuntimeError('Plan generated more than once')
        hidden=(output[0] if isinstance(output,tuple) else output).detach()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=hidden.is_cuda):
            plan=planner.plan(hidden,previous,previous_valid,mode=mode,temperature=temperature,
                keep=keep,offsets=offsets,drop_previous=drop_previous,teacher_masks=teacher_masks,
                teacher_probability=teacher_probability)
        trace['plan']=plan;trace['prefix_hidden']=hidden;trace['predictor_calls']+=1
        for step,layer in enumerate(range(c.dense_prefix,c.llm_layers)):
            trace['gates'][layer]=torch.ones_like(plan.gates[:,:,step]) if layer<dense_until else plan.gates[:,:,step]
            trace['soft'][layer]=plan.soft[:,:,step]
            trace['scores'][layer]=plan.scores[:,:,step].detach()
            controller.state.gates[layer,'mlp']=trace['gates'][layer]
        trace['events'].append(('plan_complete',c.dense_prefix))

    def final_hidden(module,args):trace['pre_norm']=args[0]

    try:
        with controller.use_gates():
            for layer_id,layer in enumerate(model.model.layers):
                if checkpoint_layers and layer_id>=c.dense_prefix:
                    originals.append((layer,layer.forward));layer.forward=checkpoint_forward(layer_id,layer.forward)
                handles.append(layer.mlp.down_proj.register_forward_pre_hook(observe(layer_id)))
            handles.append(model.model.layers[c.dense_prefix-1].register_forward_hook(make_plan))
            handles.append(model.model.norm.register_forward_pre_hook(final_hidden))
            yield trace
    finally:
        for h in handles:h.remove()
        for layer,original in originals:layer.forward=original


@torch.no_grad()
def record_parallel_paths(model,controller,planner,thresholds,ids,*,offsets=None,keep=.5,
                          attention_mask=None,drop_previous=False):
    """Bootstrap recording pass with unavailable previous history, not a fixed point."""
    with planner_route(model,controller,planner,thresholds,mode='hard',offsets=offsets,
                       keep=keep,checkpoint_layers=False,drop_previous=drop_previous) as trace:
        model.model(ids,attention_mask=attention_mask,use_cache=False)
    bits=torch.stack([trace['bits'][layer] for layer in range(planner.config.llm_layers)],dim=2)
    return previous_context(bits)
