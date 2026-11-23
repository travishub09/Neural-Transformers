"""Neuron-level, two-layer-ahead prediction and causally gated quality simulation.

No backbone gradients and no magnitude-valued inputs to the binary predictor.
The same-layer previous-token embedding is reused, not projected a second time.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from .path_model import PathConfig, PathModel, PathCache, LayerPrediction
from .sparsity_diagnostics import select_mask


def neuron_config(signal='binary', **kwargs):
    return PathConfig(units=6144, width=256, depth=2, heads=4,
                      hidden_summary=256, signal=signal, **kwargs)


def shifted(x):
    """Previous position within each document window; no wraparound."""
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)


class NeuronPredictor(PathModel):
    def _history_embed(self, states, hidden, layers):
        return self._embed(states, hidden, layers)

    def _add_current_hidden(self, x, hidden, layers):
        return x

    def _reuse_previous(self, embeddings, previous_embedding=None):
        # [B,T,S,D]; the first token has no predecessor in this context window.
        valid = torch.arange(embeddings.shape[1], device=embeddings.device)[None, :, None, None] > 0
        previous=shifted(embeddings)
        if previous_embedding is not None:
            previous[:,0]=previous_embedding
            valid=valid.expand(embeddings.shape[0],-1,-1,-1).clone();valid[:,0]=True
        return embeddings + torch.where(valid,
            self.previous_projection(previous) + self.previous_available, 0.)

    def forward_tokens(self, states=None, hidden=None):
        source = hidden if self.config.signal == 'hidden' else states
        b, t, layers, _ = source.shape
        flatten = lambda x: None if x is None else x.reshape(b*t, layers, -1)[:, :self.steps]
        x = self._history_embed(flatten(states), flatten(hidden), slice(0, self.steps))
        x = self._reuse_previous(x.reshape(b,t,self.steps,-1)).reshape(b*t,self.steps,-1)
        x = self._add_current_hidden(x, flatten(hidden), slice(0, self.steps))
        x = x + self.layer_embedding
        for block in self.blocks:
            x, _ = block(x)
        return (torch.einsum('nsd,sdu->nsu', self.final_norm(x), self.output_weight)
                + self.output_bias).reshape(b,t,self.steps,-1)

    def step_tokens(self, states=None, hidden=None, cache=None, previous_embedding=None, embedding_store=None):
        source = hidden if self.config.signal == 'hidden' else states
        b,t,_ = source.shape
        layer = 0 if cache is None else cache.next_layer
        if layer >= self.steps:
            raise ValueError('No remaining target layer')
        if cache is not None and (cache.owner != id(self) or cache.batch_size != b*t):
            raise ValueError('Mismatched predictor cache')
        expand = lambda x: None if x is None else x.reshape(b*t,1,-1)
        x = self._history_embed(expand(states), expand(hidden), slice(layer,layer+1))
        x=x.reshape(b,t,1,-1)
        if embedding_store is not None:
            embedding_store[layer]=x[:,-1].detach().clone()
        x = self._reuse_previous(x,previous_embedding).reshape(b*t,1,-1)
        x = self._add_current_hidden(x, expand(hidden), slice(layer,layer+1))
        x = x + self.layer_embedding[layer]
        updated = []
        for i,block in enumerate(self.blocks):
            x,kv = block(x, None if cache is None else cache.blocks[i]);updated.append(kv)
        logits = self.final_norm(x[:,0]) @ self.output_weight[layer] + self.output_bias[layer]
        return LayerPrediction(layer+2, logits.reshape(b,t,-1),
            PathCache(layer+1,tuple(updated),b*t,id(self)))

    def accounting(self):
        result = super().accounting()
        c=self.config
        duplicate = self.steps * (4*c.units*c.width if c.signal in ('binary','combined') else
                    2*c.hidden_size*c.hidden_summary+2*c.hidden_summary*c.width)
        if c.previous_token:
            result['matmul_flops_per_llm_token'] -= duplicate
        result['assumptions'][0] = 'Previous same-layer embedding reused across adjacent tokens; dense embedding matmuls counted'
        return result


@contextmanager
def collect_dense(model, thresholds=None, budgets=(.5,.4), sample_raw=False, capture_hidden=True):
    """Ephemeral per-batch labels/history from a normal dense backbone pass."""
    records={'labels':{},'bits':{},'hidden':{},'raw':{}}
    handles=[]
    def before(l):
        def hook(module,args):
            x=args[0].detach();a=x.float().abs()
            records['labels'][l]={str(k):select_mask(a,k) for k in budgets}
            if thresholds is not None:
                records['bits'][l]=(a>thresholds[l]).to(torch.int8)
            if sample_raw:
                records['raw'][l]=a.to(torch.bfloat16)
        return hook
    def after(l):
        def hook(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            # RMS normalization is local to this hidden vector, including at inference.
            h=h.detach().float();records['hidden'][l]=(h/h.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)).to(torch.bfloat16)
        return hook
    try:
        for l,layer in enumerate(model.model.layers):
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(before(l)))
            if capture_hidden:
                handles.append(layer.register_forward_hook(after(l)))
        yield records
    finally:
        for h in handles:h.remove()


def stack_stream(records,key):
    return torch.stack([records[key][l] for l in sorted(records[key])],dim=2)


@contextmanager
def route_neurons(model,controller,thresholds,keep,*,predictor=None,method='fixed',
                  fixed=None,teacher_masks=None,teal_thresholds=None,hide_probability=1.,
                  previous_embeddings=None,next_embeddings=None):
    """Choose gates before target layers; gather only selected-neuron observations.

    `previous_oracle` is explicitly privileged: it borrows dense previous-token
    labels. Literal previous-selected-mask propagation is the static mask seeded
    at position zero, and is reported as identical to `fixed`.
    """
    records={'bits':{},'hidden':{},'masks':{}}
    handles=[];cache=None
    def before(l):
        def hook(module,args):
            x=args[0];a=x.detach().float().abs()
            if l<2:
                mask=torch.ones_like(x,dtype=torch.bool)
            elif predictor is not None:
                mask=controller.state.gates[l,'mlp']
            elif method=='fixed':
                mask=fixed[l].expand_as(x)
            elif method=='oracle':
                mask=select_mask(a,keep)
            elif method=='dense_oracle':
                mask=teacher_masks[l]
            elif method=='previous_oracle':
                mask=shifted(teacher_masks[l])
                mask[:,0]=fixed[l]
            elif method=='wina_mlp':
                mask=select_mask(a*module.weight.detach().float().norm(dim=0),keep)
            elif method=='teal_mlp':
                mask=a>teal_thresholds[l]
            else:
                raise ValueError(method)
            if predictor is not None and l>=2 and hide_probability<1:
                mask=mask | (torch.rand_like(a)>=hide_probability)
            controller.state.gates[l,'mlp']=mask
            records['masks'][l]=mask.detach()
            records['bits'][l]=torch.where(mask,(a>thresholds[l]).to(torch.int8),-1)
        return hook
    def after(l):
        def hook(module,args,output):
            nonlocal cache
            h=output[0] if isinstance(output,tuple) else output
            h=h.detach().float();h=(h/h.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)).to(torch.bfloat16)
            records['hidden'][l]=h
            if predictor is not None and l<predictor.steps:
                kw={}
                if predictor.config.signal in ('binary','combined'): kw['states']=records['bits'][l]
                if predictor.config.signal in ('hidden','combined'): kw['hidden']=h
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=h.is_cuda):
                    prediction=predictor.step_tokens(**kw,cache=cache,
                        previous_embedding=None if previous_embeddings is None else previous_embeddings.get(l),
                        embedding_store=next_embeddings)
                cache=prediction.cache
                controller.state.gates[prediction.target_layer,'mlp']=select_mask(prediction.logits,keep)
        return hook
    try:
        for l,layer in enumerate(model.model.layers):
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(before(l)))
            handles.append(layer.register_forward_hook(after(l)))
        yield records
    finally:
        for h in handles:h.remove()


def recall_weighted_loss(logits,labels,positive_weight=2.):
    return F.binary_cross_entropy_with_logits(logits.float(),labels.float(),
        pos_weight=torch.tensor(positive_weight,device=logits.device))


def recovered_advantage(fixed,predicted,oracle):
    denominator=fixed-oracle
    return None if denominator<=1e-9 else (fixed-predicted)/denominator
