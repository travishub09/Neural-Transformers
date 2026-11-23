"""Whole-suffix planning from a dense prefix and a previous token's observations.

No current-token suffix activity is an input. Autoregressive steps describe
layers, not language tokens. Hard forward choices retain surrogate gradients
through the complete sequence of mask embeddings during quality training.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from .path_model import CausalBlock, PathConfig
from .sparsity_diagnostics import select_mask


@dataclass(frozen=True)
class PlannerConfig:
    llm_layers: int = 28
    dense_prefix: int = 2
    units: int = 6144
    hidden_size: int = 2048
    width: int = 256
    depth: int = 2
    heads: int = 4
    expansion: int = 4
    autoregressive: bool = True
    previous_path: bool = True

    def __post_init__(self):
        if not 0 < self.dense_prefix < self.llm_layers:
            raise ValueError('The dense prefix must precede a nonempty suffix')
        if min(self.units,self.hidden_size,self.width,self.depth,self.heads)<1 or self.width%self.heads:
            raise ValueError('Invalid planner dimensions')


@dataclass
class SuffixPlan:
    scores: torch.Tensor
    gates: torch.Tensor
    soft: torch.Tensor
    feedback: torch.Tensor


def choose_gates(scores, *, temperature=1., mode='hard', keep=.5):
    if temperature<=0:raise ValueError('Temperature must be positive')
    if mode not in ('hard','soft','gumbel','topk','dense'):raise ValueError(mode)
    value=scores.float()
    if mode=='gumbel':
        u=torch.rand_like(value).clamp(1e-6,1-1e-6)
        # Binary Concrete: the difference of two Gumbels is logistic noise.
        value=value+u.log()-torch.log1p(-u)
    soft=torch.sigmoid(value/temperature)
    if mode=='dense':gates=torch.ones_like(soft)
    elif mode=='soft':gates=soft
    elif mode=='topk':
        gates=select_mask(scores,keep).to(soft.dtype)+(soft-soft.detach())
    else:
        hard=(value>=0).to(soft.dtype)
        gates=hard+(soft-soft.detach())
    return gates,soft


class SuffixPlanner(nn.Module):
    def __init__(self,config=PlannerConfig()):
        super().__init__();self.config=config;c=config
        self.steps=c.llm_layers-c.dense_prefix
        self.context_projection=nn.Linear(c.hidden_size,c.width,bias=False)
        if c.previous_path:
            self.previous_tables=nn.Parameter(torch.empty(c.llm_layers,c.units,c.width))
            self.previous_available=nn.Parameter(torch.zeros(c.width))
        else:
            self.register_parameter('previous_tables',None)
            self.register_parameter('previous_available',None)
        if c.autoregressive:
            self.mask_tables=nn.Parameter(torch.empty(self.steps-1,c.units,c.width))
        else:self.register_parameter('mask_tables',None)
        self.layer_embedding=nn.Parameter(torch.empty(self.steps,c.width))
        body=PathConfig(llm_layers=c.llm_layers,units=c.units,width=c.width,depth=c.depth,
            heads=c.heads,expansion=c.expansion,lookahead=c.dense_prefix)
        self.blocks=nn.ModuleList([CausalBlock(body) for _ in range(c.depth)])
        self.final_norm=nn.LayerNorm(c.width)
        self.output_weight=nn.Parameter(torch.empty(self.steps,c.width,c.units))
        self.output_bias=nn.Parameter(torch.zeros(self.steps,c.units))
        nn.init.normal_(self.context_projection.weight,std=1/math.sqrt(c.hidden_size))
        nn.init.normal_(self.layer_embedding,std=.02)
        nn.init.normal_(self.output_weight,std=1/math.sqrt(c.width))
        if self.mask_tables is not None:nn.init.normal_(self.mask_tables,std=1/math.sqrt(c.units))
        if self.previous_tables is not None:
            nn.init.normal_(self.previous_tables,std=1/math.sqrt(c.llm_layers*c.units))

    def context(self,hidden,previous=None,previous_valid=None,*,drop_previous=False):
        c=self.config
        if hidden.ndim!=3 or hidden.shape[-1]!=c.hidden_size or not torch.isfinite(hidden).all():
            raise ValueError('Expected finite dense-prefix hidden states [batch,tokens,hidden]')
        # Identical nonlearned RMS operation as the closed-loop content baseline.
        h=hidden.float();h=h/h.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
        x=self.context_projection(h.to(self.context_projection.weight.dtype))
        if previous is None:
            if previous_valid is not None and previous_valid.any():
                raise ValueError('A valid previous token needs observations')
            return x
        if not c.previous_path:raise ValueError('Hidden-only planner must not receive path observations')
        if previous.shape!=(*hidden.shape[:2],c.llm_layers,c.units):
            raise ValueError('Previous path must be [batch,tokens,layers,neurons]')
        if previous.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64):
            raise TypeError('Previous path contains discrete observations, never magnitudes')
        if ((previous < -1)|(previous > 1)).any():raise ValueError('Invalid observation code')
        if previous_valid is None or previous_valid.shape!=hidden.shape[:2] or previous_valid.dtype!=torch.bool:
            raise ValueError('Provide explicit per-token previous-context validity')
        if drop_previous:return x
        # Only observed strong units contribute; -1 (unknown) and 0 (weak) do
        # not. There is no separate per-neuron observed-flag table in this design.
        active=((previous==1)&previous_valid[...,None,None]).to(x.dtype)
        px=torch.einsum('ntlu,lud->ntd',active,self.previous_tables)
        return x+px+previous_valid[...,None]*self.previous_available

    def _body(self,x,cache=None):
        updated=[]
        for i,block in enumerate(self.blocks):
            x,kv=block(x,None if cache is None else cache[i]);updated.append(kv)
        return x,tuple(updated)

    def teacher_forced(self,hidden,teacher_masks,previous=None,previous_valid=None,*,drop_previous=False):
        """Parallel causal teacher forcing; target i cannot see teacher mask i."""
        b,t,_=hidden.shape;c=self.config
        if teacher_masks.shape!=(b,t,self.steps,c.units):raise ValueError('Teacher mask shape mismatch')
        context=self.context(hidden,previous,previous_valid,drop_previous=drop_previous).reshape(b*t,1,c.width)
        if c.autoregressive:
            masks=teacher_masks[:,:,:-1].reshape(b*t,self.steps-1,c.units).to(context.dtype)
            tail=torch.einsum('nsu,sud->nsd',masks,self.mask_tables)
            x=torch.cat((context,tail),dim=1)
        else:
            # All queries are known up front. No predicted or teacher mask enters
            # the one-shot model, although its body can mix layer-position queries.
            x=context.expand(-1,self.steps,-1)
        x,_=self._body(x+self.layer_embedding)
        logits=torch.einsum('nsd,sdu->nsu',self.final_norm(x),self.output_weight)+self.output_bias
        return logits.reshape(b,t,self.steps,c.units)

    def plan(self,hidden,previous=None,previous_valid=None,*,mode='hard',temperature=1.,
             keep=.5,offsets=None,teacher_masks=None,teacher_probability=0.,drop_previous=False,
             score_prior=None,dense_tokens=None):
        b,t,_=hidden.shape;c=self.config
        if not 0<=teacher_probability<=1:raise ValueError('Invalid teacher probability')
        if teacher_probability and teacher_masks is None:raise ValueError('Teacher feedback requires masks')
        if teacher_masks is not None and teacher_masks.shape!=(b,t,self.steps,c.units):
            raise ValueError('Teacher mask shape mismatch')
        if offsets is None:offsets=hidden.new_zeros(self.steps)
        if offsets.shape!=(self.steps,):raise ValueError('Offsets must have one value per suffix layer')
        if score_prior is not None and score_prior.shape!=(b,t,self.steps,c.units):raise ValueError('Prior shape mismatch')
        if dense_tokens is not None and (dense_tokens.shape!=(b,t) or dense_tokens.dtype!=torch.bool):raise ValueError('Dense token mask mismatch')
        if not c.autoregressive:
            # A zero placeholder is shape-only; teacher_forced never reads it here.
            placeholder=hidden.new_empty(1).expand(b,t,self.steps,c.units)
            scores=self.teacher_forced(hidden,placeholder,previous,previous_valid,drop_previous=drop_previous)
            if score_prior is not None:scores=scores+score_prior.to(scores.dtype)
            gates,soft=choose_gates(scores-offsets[None,None,:,None],mode=mode,temperature=temperature,keep=keep)
            if dense_tokens is not None:gates=torch.where(dense_tokens[:,:,None,None],torch.ones_like(gates),gates)
            return SuffixPlan(scores,gates,soft,gates)
        context=self.context(hidden,previous,previous_valid,drop_previous=drop_previous).reshape(b*t,1,c.width)
        cache=None;scores_all=[];gates_all=[];soft_all=[];feedback_all=[]
        feedback=None
        for step in range(self.steps):
            if step==0:x=context
            else:
                # Keep this differentiable: future decisions must receive
                # gradients through earlier straight-through mask choices.
                x=(feedback.reshape(b*t,c.units).to(context.dtype)@self.mask_tables[step-1]).unsqueeze(1)
            x,cache=self._body(x+self.layer_embedding[step],cache)
            scores=(self.final_norm(x[:,0])@self.output_weight[step]+self.output_bias[step]).reshape(b,t,c.units)
            if score_prior is not None:scores=scores+score_prior[:,:,step].to(scores.dtype)
            gates,soft=choose_gates(scores-offsets[step],mode=mode,temperature=temperature,keep=keep)
            if dense_tokens is not None:gates=torch.where(dense_tokens[:,:,None],torch.ones_like(gates),gates)
            feedback=gates
            if teacher_probability:
                use_teacher=torch.rand((b,t,1),device=hidden.device)<teacher_probability
                feedback=torch.where(use_teacher,teacher_masks[:,:,step].to(gates.dtype),gates)
            scores_all.append(scores);gates_all.append(gates);soft_all.append(soft);feedback_all.append(feedback)
        return SuffixPlan(*(torch.stack(v,dim=2) for v in (scores_all,gates_all,soft_all,feedback_all)))

    def accounting(self):
        c=self.config;d=c.width;s=self.steps
        parameters=sum(p.numel() for p in self.parameters())
        context=2*c.hidden_size*d
        if c.previous_path:context+=2*c.llm_layers*c.units*d
        mask=2*(s-1)*c.units*d if c.autoregressive else 0
        body=s*c.depth*(8*d*d+4*c.expansion*d*d)
        attention=c.depth*sum(4*(i+1)*d for i in range(s))
        return {'config':asdict(c),'parameters':parameters,'bf16_weight_payload_bytes':2*parameters,
            'dense_matmul_flops_per_token':context+mask+body+attention+2*s*d*c.units,
            'scope':'Whole-plan dense projection accounting; not measured DRAM traffic or prefetch speedup'}
