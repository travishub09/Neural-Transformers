"""Observed-only, causal sequence statistics and a 52-parameter output prior.

Prompt energies are exact. Sparse-token denominators use observed units only;
uncomputed activations must never affect either numerator or denominator.
The statistic is a weighted per-observation mean of squared relative activation,
followed by log-RMS and within-layer standardization. No tracker is learned.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .open_loop import SuffixPlanner,PlannerConfig


@dataclass(frozen=True)
class TrackerConfig:
    prompt_weight: float = 4.
    epsilon: float = 1e-12
    standardization_floor: float = 1e-6
    clip: float = 8.


def relative_energy(activation,observed):
    if activation.shape!=observed.shape or observed.dtype!=torch.bool:raise ValueError('Observation shape/type mismatch')
    value=torch.where(observed,activation.detach().float(),0.)
    square=value.square()
    return square/square.sum(-1,keepdim=True).clamp_min(1e-12)


def normalize_scores(total,count,config=TrackerConfig()):
    mean=total/count.clamp_min(1.)
    log=.5*mean.clamp_min(config.epsilon).log()
    available=count>0
    # Unobserved entries are neutral, never interpreted as observed zeros.
    n=available.sum(-1,keepdim=True).clamp_min(1)
    center=(log*available).sum(-1,keepdim=True)/n
    deviation=torch.where(available,log-center,0.)
    std=(deviation.square().sum(-1,keepdim=True)/n).sqrt().clamp_min(config.standardization_floor)
    return (deviation/std).clamp(-config.clip,config.clip)


class SequenceTracker:
    def __init__(self,batch,layers,units,*,device,config=TrackerConfig()):
        self.config=config
        self.total=torch.zeros(batch,layers,units,device=device,dtype=torch.float32)
        self.count=torch.zeros_like(self.total)

    def scores(self):return normalize_scores(self.total,self.count,self.config)

    @torch.no_grad()
    def update(self,energy,observed,*,prompt=False,valid=None):
        # Inputs [B,T,L,U], supporting an exact dense prefill in one update.
        if energy.ndim!=4 or energy.shape!=observed.shape:raise ValueError('Expected [B,T,L,U]')
        if valid is None:valid=torch.ones(energy.shape[:2],dtype=torch.bool,device=energy.device)
        weight=self.config.prompt_weight if prompt else 1.
        present=observed & valid[:,:,None,None]
        self.total.add_((torch.where(present,energy.detach().float(),0.)*weight).sum(1))
        self.count.add_((present.float()*weight).sum(1))

    def accounting(self):return {'state_bytes_per_sequence':self.total[0].numel()*8,'learned_parameters':0}


@torch.no_grad()
def causal_scores(energy,observed,prompt_tokens,config=TrackerConfig()):
    """Exclusive prefix statistics: position t can only use observations <t."""
    if energy.ndim!=4 or energy.shape!=observed.shape:raise ValueError('Expected [B,T,L,U]')
    weight=torch.where(torch.arange(energy.shape[1],device=energy.device)<prompt_tokens,config.prompt_weight,1.)
    weight=weight[None,:,None,None]
    weighted=torch.where(observed,energy.detach().float(),0.)*weight
    counts=observed.float()*weight
    total=torch.cat((torch.zeros_like(weighted[:,:1]),weighted[:,:-1].cumsum(1)),1)
    count=torch.cat((torch.zeros_like(counts[:,:1]),counts[:,:-1].cumsum(1)),1)
    return normalize_scores(total,count,config)


class SequencePriorPlanner(SuffixPlanner):
    def __init__(self,config=PlannerConfig(),*,enabled=True):
        super().__init__(config)
        self.prior_gain=nn.Parameter(torch.zeros(self.steps),requires_grad=enabled)
        self.prior_bias=nn.Parameter(torch.zeros(self.steps),requires_grad=enabled)
        self.prior_enabled=enabled

    def plan(self,hidden,previous=None,previous_valid=None,*,sequence_scores=None,**kwargs):
        if sequence_scores is None:
            sequence_scores=hidden.new_zeros(*hidden.shape[:2],self.steps,self.config.units)
        if sequence_scores.shape[-2]==self.config.llm_layers:
            sequence_scores=sequence_scores[:,:,self.config.dense_prefix:]
        if self.prior_enabled:
            prior=(sequence_scores.detach().float()*self.prior_gain[None,None,:,None]
                   +self.prior_bias[None,None,:,None])
            kwargs['score_prior']=prior
        return super().plan(hidden,previous,previous_valid,**kwargs)

    @torch.no_grad()
    def initialize_from(self,state,*,prior_first=False):
        result=self.load_state_dict(state,strict=False)
        assert set(result.missing_keys)=={'prior_gain','prior_bias'} and not result.unexpected_keys
        self.prior_gain.fill_(1. if prior_first else 0.);self.prior_bias.zero_()
        if prior_first:
            self.output_weight.zero_();self.output_bias.zero_()
