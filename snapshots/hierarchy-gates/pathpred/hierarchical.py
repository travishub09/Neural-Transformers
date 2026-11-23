"""Hidden-state MLP hierarchy. All attention/KV computation stays dense.

The new coarse gates have exact-one hard forwards at initialization, with a
sigmoid straight-through derivative. This is a quality simulation: multiplying
activations after dense matmuls does not itself save wall-clock compute.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .neuron_prediction import NeuronPredictor
from .path_model import PathCache
from .end_to_end import neuron_gates


@dataclass(frozen=True)
class HierarchyPrediction:
    target_layer: int
    logits: torch.Tensor
    layer_logits: torch.Tensor | None
    group_logits: torch.Tensor | None
    cache: PathCache


class HierarchicalPredictor(NeuronPredictor):
    def __init__(self, config, variant='D', group_size=128):
        if config.signal != 'hidden' or variant not in 'ABCD' or len(variant) != 1:
            raise ValueError('Hierarchy uses a hidden-only predictor and variant A/B/C/D')
        if config.units % group_size:
            raise ValueError('Groups must evenly partition the unchanged neuron order')
        super().__init__(config)
        self.variant, self.group_size = variant, group_size
        self.groups = config.units // group_size
        for name, count in [('layer', 1 if variant in 'BD' else 0),
                            ('group', self.groups if variant in 'CD' else 0)]:
            self.register_parameter(name+'_weight', nn.Parameter(torch.zeros(self.steps, config.width, count)) if count else None)
            self.register_parameter(name+'_bias', nn.Parameter(torch.full((self.steps, count), .25)) if count else None)

    def load_hidden_state(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed = {name for name in ('layer_weight','layer_bias','group_weight','group_bias')
                   if getattr(self, name) is not None}
        if set(missing) != allowed or unexpected:
            raise ValueError(f'Warm-start mismatch: missing={missing}, unexpected={unexpected}')

    def step_tokens(self, states=None, hidden=None, cache=None, previous_embedding=None, embedding_store=None):
        if states is not None or hidden is None:
            raise ValueError('Hierarchy cannot read path observations')
        b,t,_ = hidden.shape
        layer = 0 if cache is None else cache.next_layer
        if layer >= self.steps or (cache is not None and (cache.owner != id(self) or cache.batch_size != b*t)):
            raise ValueError('Mismatched hierarchy cache')
        x = self._history_embed(None, hidden.reshape(b*t,1,-1), slice(layer,layer+1))
        x = x.reshape(b,t,1,-1)
        if embedding_store is not None:
            embedding_store[layer] = x[:,-1].detach().clone()
        x = self._reuse_previous(x, previous_embedding).reshape(b*t,1,-1)
        x = x + self.layer_embedding[layer]
        updated=[]
        for i,block in enumerate(self.blocks):
            x,kv=block(x, None if cache is None else cache.blocks[i]);updated.append(kv)
        x=self.final_norm(x[:,0])
        logits=(x @ self.output_weight[layer]+self.output_bias[layer]).reshape(b,t,-1)
        def coarse(name):
            weight=getattr(self,name+'_weight')
            return None if weight is None else (x @ weight[layer]+getattr(self,name+'_bias')[layer]).reshape(b,t,-1)
        return HierarchyPrediction(layer+self.config.lookahead, logits, coarse('layer'), coarse('group'),
            PathCache(layer+1,tuple(updated),b*t,id(self)))

    def accounting(self):
        result=super().accounting()
        count=(1 if self.layer_weight is not None else 0)+(self.groups if self.group_weight is not None else 0)
        result['matmul_flops_per_llm_token'] += 2*self.steps*self.config.width*count
        result.update(variant=self.variant, group_size=self.group_size,
                      bf16_weight_bytes=sum(p.numel() for p in self.parameters())*2)
        return result


def effective_gates(prediction, group_size, *, offset=0., global_shift=0., temperature=.3, mode='hard'):
    neuron,ns,_=neuron_gates(prediction.logits-offset-global_shift,temperature,mode)
    shape=neuron.shape[:-1]+(1,)
    def coarse(logits):
        if logits is None:return neuron.new_ones(shape),neuron.new_ones(shape)
        g,s,_=neuron_gates(logits-global_shift,temperature,mode);return g,s
    layer,ls=coarse(prediction.layer_logits)
    group,gs=coarse(prediction.group_logits)
    if group.shape[-1] != 1:
        group=group.repeat_interleave(group_size,dim=-1)
        gs=gs.repeat_interleave(group_size,dim=-1)
    return {'effective':layer*group*neuron,'soft':ls*gs*ns,
            'layer':layer,'group':group,'neuron':neuron}


@dataclass(frozen=True)
class DecodeCost:
    """Multiply-add=2 FLOPs, batch-1 decode at a declared reference context.

    Denominator is dense Qwen's attention, MLP and LM-head matmuls. Numerator
    adds the predictor's dense matmul estimate. Norms/activations/launches are
    excluded consistently. The observed dense training runtime is separate.
    """
    layers: int
    prefix: int
    units: int
    neuron_flops: int
    attention_flops: int
    lm_head_flops: int
    predictor_flops: int
    reference_context: int

    @classmethod
    def from_model(cls, model, predictor, reference_context=1024):
        c=model.config;h=c.hidden_size;qd=c.num_attention_heads*c.head_dim;kv=c.num_key_value_heads*c.head_dim
        return cls(c.num_hidden_layers,predictor.config.lookahead,c.intermediate_size,6*h,
            4*h*(qd+kv)+4*qd*reference_context,2*h*c.vocab_size,
            predictor.accounting()['matmul_flops_per_llm_token'],reference_context)

    @property
    def dense(self):return self.layers*(self.attention_flops+self.units*self.neuron_flops)+self.lm_head_flops
    @property
    def fixed(self):return self.layers*self.attention_flops+self.prefix*self.units*self.neuron_flops+self.lm_head_flops+self.predictor_flops
    @property
    def variable(self):return (self.layers-self.prefix)*self.units*self.neuron_flops
    def fraction(self, retained):return (self.fixed+self.variable*retained)/self.dense
    def retained_for(self, target):
        result=(target*self.dense-self.fixed)/self.variable
        if not 0 < result < 1:raise ValueError('Requested total compute is infeasible with dense attention/prefix')
        return result


def gate_accounting(trace, prefix=2, scored_only=False):
    """Exclusive savings: layer first, then group within open layers, then neuron.

    Reversing the attribution order changes the attribution but never total cost.
    """
    values={k:[] for k in ('used','layer_saved','group_saved','neuron_saved','layer_off','group_off',
                          'redundant_group_off','redundant_neuron_off')}
    for l,c in trace['components'].items():
        if l<prefix:continue
        n,g,a=c['neuron'],c['group'],c['layer']
        cut=lambda x:x[:,:-1] if scored_only else x
        values['used'].append(cut(c['effective']).mean(-1))
        values['layer_saved'].append(cut(1-a).squeeze(-1))
        values['group_saved'].append(cut(a*(1-g)).expand_as(cut(n)).mean(-1))
        values['neuron_saved'].append(cut(a*g*(1-n)).mean(-1))
        values['layer_off'].append(cut((a.detach()<.5).float()).squeeze(-1))
        values['group_off'].append(cut((g.detach()<.5).float()).expand_as(cut(n)).mean(-1))
        values['redundant_group_off'].append(cut((1-a)*(1-g)).expand_as(cut(n)).mean(-1))
        values['redundant_neuron_off'].append(cut((1-a*g)*(1-n)).mean(-1))
    return {k:torch.stack(v,dim=2) for k,v in values.items()}


@contextmanager
def hierarchy_route(model,controller,predictor,thresholds,*,temperature=.3,mode='hard',
                    checkpoint_layers=True,score_offsets=None,global_shift=0.):
    if controller.state.active:raise RuntimeError('hierarchy_route owns gates')
    trace={'bits':{},'gates':{},'soft':{},'scores':{},'components':{},'pre_norm':None,
           'predictor_calls':0,'recomputations':0}
    handles=[];originals=[];cache=None;recomputing=False
    @contextmanager
    def recompute():
        nonlocal recomputing
        old=recomputing;recomputing=True;trace['recomputations']+=1
        try:yield
        finally:recomputing=old
    def checkpoint_forward(i,original):
        def wrapped(hidden,*args,**kwargs):
            gate=controller.state.gates.get((i,'mlp'))
            if gate is None or not torch.is_grad_enabled():return original(hidden,*args,**kwargs)
            def run(h,g):
                if not controller.state.active:raise RuntimeError('Backward must remain inside route')
                controller.state.gates[i,'mlp']=g;return original(h,*args,**kwargs)
            return checkpoint(run,hidden,gate,use_reentrant=False,context_fn=lambda:(nullcontext(),recompute()))
        return wrapped
    def observe(i):
        def hook(module,args):
            if recomputing:return
            a=args[0].detach().float().abs()
            if i<predictor.config.lookahead:trace['gates'][i]=torch.ones_like(a,dtype=torch.bool)
            observed=trace['gates'][i].detach()>=.5
            trace['bits'][i]=torch.where(observed,(a>thresholds[i]).to(torch.int8),-1)
        return hook
    def predict(i):
        def hook(module,args,output):
            nonlocal cache
            if i>=predictor.steps:return
            h=(output[0] if isinstance(output,tuple) else output).detach().float()
            h=(h/h.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)).to(torch.bfloat16)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=h.is_cuda):
                p=predictor.step_tokens(hidden=h,cache=cache)
            cache=p.cache;l=p.target_layer
            offset=0. if score_offsets is None else score_offsets[l]
            c=effective_gates(p,predictor.group_size,offset=offset,global_shift=global_shift,temperature=temperature,mode=mode)
            trace['components'][l]=c;trace['gates'][l]=c['effective'];trace['soft'][l]=c['soft']
            trace['scores'][l]=p.logits.detach();trace['predictor_calls']+=1
            controller.state.gates[l,'mlp']=c['effective']
        return hook
    def final_hidden(module,args):trace['pre_norm']=args[0]
    try:
        with controller.use_gates():
            for i,layer in enumerate(model.model.layers):
                if checkpoint_layers:
                    originals.append((layer,layer.forward));layer.forward=checkpoint_forward(i,layer.forward)
                handles.append(layer.mlp.down_proj.register_forward_pre_hook(observe(i)))
                handles.append(layer.register_forward_hook(predict(i)))
            handles.append(model.model.norm.register_forward_pre_hook(final_hidden))
            yield trace
    finally:
        for h in handles:h.remove()
        for layer,original in originals:layer.forward=original
