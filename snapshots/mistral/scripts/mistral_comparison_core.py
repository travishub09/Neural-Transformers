"""Shared causal evaluation; no training or modifications to producing model code."""
from contextlib import contextmanager
import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from pathpred.planner_route import planner_route


def griffin_scores(activation,valid):
    # Official GRIFFIN: L2 norm across prompt positions of per-token L2-normalized activations.
    x=activation.float();relative=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return (relative.square()*valid[...,None]).sum(1).sqrt()


def top_mask(scores,keep):
    k=int(scores.shape[-1]*keep)
    return torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,scores.topk(k,dim=-1).indices,True)


def attention_sketch(weights,valid):
    p=weights[:,:,0];length=valid.sum(-1);left=valid.shape[-1]-length
    # Prompts are left padded, so centroid features use the actual unpadded key positions.
    pmax=p.max(-1).values
    arg=(p.argmax(-1)-left[:,None]).float()/(length-1).clamp_min(1)[:,None]
    ent=-(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum(-1)
    top4=p.topk(min(4,p.shape[-1]),dim=-1).values.sum(-1)
    return torch.stack((pmax,arg,ent,top4),dim=-1).flatten(1)


class ReleasedNeuro:
    def __init__(self,reference,device='cuda',cap=7168):
        reference=Path(reference);sys.path.insert(0,str(reference/'quality_study'))
        from np_python_v2.inproc_predictor import Predictor,_infer_meta_from_state_dict
        from np_python_v2.centroid_bank import _extract_tensor,_to_binary
        assets=reference/'assets'
        sd=torch.load(assets/'gate_up_predictor_mistral_30p_d4_t512_k512.pt',map_location='cpu',weights_only=True,mmap=True)
        pattern,vocab=_infer_meta_from_state_dict(sd)
        self.net=Predictor(pattern_dim=pattern,g0_vocab=vocab,depth=4).eval()
        self.net.load_state_dict(sd,strict=True);del sd
        self.net.to(device=device,dtype=torch.float16);self.cap=cap
        self.gate=_to_binary(_extract_tensor(torch.load(assets/'centroids_gate_512_sparsity_30.pt',map_location='cpu',weights_only=True))).to(device).bool()
        self.up=_to_binary(_extract_tensor(torch.load(assets/'centroids_up_512_sparsity_30.pt',map_location='cpu',weights_only=True))).to(device).bool()
        self.threshold=json.loads((assets/'thresholds_50_percent_sparsity.json').read_text())['gate_proj']['layer_0']
        self.last_ids=None

    def predict(self,E,P,H,N,G):
        binary=G>self.threshold
        # Preserve the released engine's overlap rule, including fp16 accumulation.
        overlap=binary.half()@self.gate[:self.net.g0_emb[0].num_embeddings].half().T
        g0=(binary.sum(-1,keepdim=True)-overlap).argmin(-1)
        gl,ul=self.net(E.half(),P.half(),H.half(),N.half(),g0)
        gi,ui=gl.argmax(-1),ul.argmax(-1);self.last_ids=(gi,ui)
        mask=self.gate[gi]&self.up[ui]
        # Official engine keeps the first active neuron IDs up to kmax, not magnitude top-k.
        return mask&(mask.cumsum(-1)<=self.cap)

    def reset_features(self):self.feature_cache=None

    def initialize_features(self,model,batch):
        from engine.decode_engine_a100 import KVCache,build_rope_cache
        layer=model.model.layers[0]
        self.feature_weights={
            'attn_q_proj':layer.self_attn.q_proj.weight,'attn_k_proj':layer.self_attn.k_proj.weight,
            'attn_v_proj':layer.self_attn.v_proj.weight,'attn_o_proj':layer.self_attn.o_proj.weight,
            'input_layernorm':layer.input_layernorm.weight,'post_attention_layernorm':layer.post_attention_layernorm.weight,
            'mlp_gate':layer.mlp.gate_proj.weight}
        self.feature_cache=[KVCache(1,8,128,4096) for _ in range(batch)]
        self.feature_rope=build_rope_cache(4096,128,theta=model.config.rope_theta)

    def prefill_features(self,model,ids,valid):
        from engine.decode_engine_a100 import rms_norm,apply_rope
        self.initialize_features(model,ids.shape[0]);w=self.feature_weights
        for b in range(ids.shape[0]):
            cache=self.feature_cache[b]
            for tid in ids[b,valid[b]]:
                x=rms_norm(model.model.embed_tokens.weight[tid],w['input_layernorm'])
                k=(w['attn_k_proj']@x).view(8,128);v=(w['attn_v_proj']@x).view(8,128)
                k=apply_rope(k,*self.feature_rope,cache.seq_len)
                cache.update(0,k[:,None],v[:,None]);cache.advance()

    def features(self,model,ids):
        from engine.decode_engine_a100 import rms_norm,attention_forward,head_attn_sketch
        if getattr(self,'feature_cache',None) is None:self.initialize_features(model,ids.shape[0])
        result=[];w=self.feature_weights
        for b in range(ids.shape[0]):
            cache=self.feature_cache[b];E=model.model.embed_tokens.weight[ids[b,0]]
            P=rms_norm(E,w['input_layernorm'])
            attn,weights=attention_forward(P,w,cache,0,cache.seq_len,*self.feature_rope,return_attn_weights=True)
            post=rms_norm(E+attn,w['post_attention_layernorm'])
            N=post.float().square().mean().sqrt().reshape(1);G=w['mlp_gate']@post
            result.append((E.float(),P.float(),head_attn_sketch(weights).float(),N,G.float()));cache.advance()
        return [torch.stack([r[i] for r in result]) for i in range(5)]

    @contextmanager
    def route(self,model,ctrl,prefix,valid,ids):
        # Released features are independent of suffix execution and extracted with
        # the release's exact first-layer arithmetic. Backbone scoring remains shared.
        masks=self.predict(*self.features(model,ids))
        gates={l:masks[:,l-1:l] for l in range(prefix,len(model.model.layers))}
        with ctrl.use_gates({(l,'mlp'):g for l,g in gates.items()}):yield {'gates':gates}


class Policy:
    def __init__(self,model,ctrl,kind,*,prefix=2,keep=.5,planner=None,q=None,offsets=None,neuro=None):
        self.model=model;self.ctrl=ctrl;self.kind=kind;self.prefix=prefix;self.keep=keep
        self.planner=planner;self.q=q;self.offsets=offsets;self.neuro=neuro
        self.reset()
    def reset(self):
        self.cache=None;self.previous=None;self.fixed={};self.last_gates={}
        self.reuse_sum=None;self.active_sum=None;self.previous_masks=None
        if self.kind=='neuro':self.neuro.reset_features()
    @torch.no_grad()
    def prefill(self,ids,valid):
        handles=[];bits={}
        if self.kind=='neuro':self.neuro.prefill_features(self.model,ids,valid)
        def collect(l):
            def hook(m,a):
                if self.kind=='griffin' and l>=self.prefix:
                    self.fixed[l,'mlp']=top_mask(griffin_scores(a[0],valid),self.keep)[:,None]
                if self.kind=='planner':bits[l]=(a[0][:,-1:].float().abs()>self.q[l]).to(torch.int8)
            return hook
        try:
            if self.kind in ('griffin','planner'):
                handles=[x.mlp.down_proj.register_forward_pre_hook(collect(l)) for l,x in enumerate(self.model.model.layers)]
            with self.ctrl.use_gates():
                out=self.model(ids,attention_mask=valid,position_ids=(valid.long().cumsum(-1)-1).clamp_min(0),use_cache=True)
            self.cache=out.past_key_values
            if bits:self.previous=torch.stack([bits[l] for l in range(len(bits))],dim=2)
            return out.logits[:,-1].float()
        finally:
            for h in handles:h.remove()
    @torch.no_grad()
    def step(self,ids,valid):
        kwargs=dict(attention_mask=valid,past_key_values=self.cache,use_cache=True,
            position_ids=(valid.sum(-1)-1)[:,None])
        if self.kind=='planner':
            present=None if self.previous is None else torch.ones(ids.shape[0],1,dtype=torch.bool,device=ids.device)
            with planner_route(self.model,self.ctrl,self.planner,self.q,previous=self.previous,previous_valid=present,
                    offsets=self.offsets,mode='hard',keep=self.keep,checkpoint_layers=False,execution_prefix=self.prefix) as tr:
                out=self.model(ids,**kwargs)
            self.previous=torch.stack([tr['bits'][l] for l in range(len(self.model.model.layers))],dim=2)
            self.last_gates={l:g for l,g in tr['gates'].items() if l>=self.prefix}
        elif self.kind=='neuro':
            with self.neuro.route(self.model,self.ctrl,self.prefix,valid,ids) as tr:out=self.model(ids,**kwargs)
            self.last_gates=tr['gates']
        else:
            with self.ctrl.use_gates(self.fixed if self.kind=='griffin' else {}):out=self.model(ids,**kwargs)
            self.last_gates={l:g for (l,k),g in self.fixed.items()}
        self.cache=out.past_key_values
        return out.logits[:,0].float()
    def masks(self):
        if not self.last_gates:return None
        return torch.stack([self.last_gates[l][:,0]>=.5 for l in range(self.prefix,len(self.model.model.layers))],1)


@torch.no_grad()
def score_batch(policy,sequences,prompt_lengths,*,teacher=True,progress=None):
    """All prompt tokens dense. Score suffix only; each row has independent state/cache.

    First continuation token is predicted by dense prompt logits for every method.
    Its inclusion in likelihood is standard generation; retention counts only sparse executions.
    Zero prompt is supported for the original all-token WikiText/document protocol.
    """
    if not sequences or len(sequences)!=len(prompt_lengths):raise ValueError('Invalid batch')
    if any(p<0 or p>=len(s) for s,p in zip(sequences,prompt_lengths)):raise ValueError('Invalid prompt')
    if any(prompt_lengths) and not all(prompt_lengths):raise ValueError('Do not mix absent and present prompts')
    if policy.kind=='griffin' and not all(prompt_lengths):raise ValueError('GRIFFIN needs a declared prompt')
    device=next(policy.model.parameters()).device;b=len(sequences);policy.reset()
    dense=Policy(policy.model,policy.ctrl,'dense') if teacher else None
    sums=torch.zeros(b,5,device=device,dtype=torch.float64)
    keep_sum=torch.zeros(b,device=device,dtype=torch.float64);executions=torch.zeros_like(keep_sum)
    reused=torch.zeros_like(keep_sum);selected=torch.zeros_like(keep_sum);previous=None
    if all(prompt_lengths):
        width=max(prompt_lengths);prompt=torch.zeros(b,width,device=device,dtype=torch.long);valid=torch.zeros_like(prompt,dtype=torch.bool)
        for i,(seq,p) in enumerate(zip(sequences,prompt_lengths)):
            prompt[i,-p:]=torch.tensor(seq[:p],device=device);valid[i,-p:]=True
        z=policy.prefill(prompt,valid);a=dense.prefill(prompt,valid) if dense else None
        suffix=[s[p:] for s,p in zip(sequences,prompt_lengths)]
        start=0
    else:
        valid=torch.empty(b,0,device=device,dtype=torch.bool);suffix=sequences;start=1;z=a=None
    maximum=max(map(len,suffix))
    for t in range(start,maximum):
        live=torch.tensor([t<len(s) for s in suffix],device=device)
        if t>0 or not all(prompt_lengths):
            current=torch.tensor([[s[t-1] if t-1<len(s) else 0] for s in suffix],device=device)
            valid=torch.cat((valid,torch.ones(b,1,device=device,dtype=torch.bool)),1)
            z=policy.step(current,valid);a=dense.step(current,valid) if dense else None
            masks=policy.masks()
            if masks is not None:
                keep_sum+=masks.double().mean((1,2))*live;executions+=live
                if previous is not None:
                    reused+=(masks&previous).sum((1,2))*live;selected+=masks.sum((1,2))*live
                previous=masks
        target=torch.tensor([s[t] if t<len(s) else 0 for s in suffix],device=device)
        lp=F.log_softmax(z.float(),-1);nll=-lp.gather(-1,target[:,None])[:,0]
        if dense:
            tp=F.log_softmax(a.float(),-1)
            vals=torch.stack((-tp.gather(-1,target[:,None])[:,0],nll,(tp.exp()*(tp-lp)).sum(-1),(tp.argmax(-1)==lp.argmax(-1)).float(),torch.ones(b,device=device)),1)
        else:vals=torch.stack((torch.zeros_like(nll),nll,torch.zeros_like(nll),torch.zeros_like(nll),torch.ones_like(nll)),1)
        sums+=vals.double()*live[:,None]
        if progress and t%32==0:progress(t,maximum)
    return {'sums':sums.cpu(),'kept_sum':keep_sum.cpu(),'executions':executions.cpu(),
        'reused':reused.cpu(),'selected_for_reuse':selected.cpu()}
