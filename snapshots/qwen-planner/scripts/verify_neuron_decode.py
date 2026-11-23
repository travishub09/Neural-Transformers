"""Measure BF16 full-sequence versus cached-decode drift on validation text.

Hard top-k gates can amplify rounding differences; report differences instead of
assuming that an FP32 causal-structure test proves BF16 mask equality.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from pathpred.collection import atomic_json
from pathpred.neuron_prediction import NeuronPredictor,neuron_config,route_neurons,stack_stream
from scripts.oracle_quality import select_documents
from scripts.neuron_guessing import ROOT,CAL,cfg


@torch.no_grad()
def verify(model,controller):
    records=[json.loads(line) for line in Path('data/stage3-corpus-v1/sequences.jsonl').read_text().splitlines()]
    selected=select_documents(records,'validation',1,cfg())
    tokens=load_file('data/stage3-corpus-v1/tokens.safetensors')['input_ids']
    cal=load_file(str(CAL),device='cuda');results=[]
    for signal in ['dense','binary','hidden']:
        p=None
        if signal!='dense':
            p=NeuronPredictor(neuron_config(signal)).cuda().eval()
            p.load_state_dict(load_file(str(ROOT/f'{signal}.safetensors'),device='cuda'))
        for keep in ([1.] if p is None else [.5,.4]):
            for meta in selected:
                ids=tokens[meta['sequence_id']:meta['sequence_id']+1,:16].long().cuda()
                with controller.use_gates():
                    if p is None:
                        full=model(ids,use_cache=False).logits
                    else:
                        with route_neurons(model,controller,cal['q75'],keep,predictor=p) as history:
                            full=model(ids,use_cache=False).logits
                        full_masks=stack_stream(history,'masks');del history
                cache=None;previous=None;parts=[];equal=total=0
                for t in range(ids.shape[1]):
                    next_embeddings={}
                    with controller.use_gates():
                        if p is None:
                            out=model(ids[:,t:t+1],past_key_values=cache,use_cache=True)
                        else:
                            with route_neurons(model,controller,cal['q75'],keep,predictor=p,
                                previous_embeddings=previous,next_embeddings=next_embeddings) as part:
                                out=model(ids[:,t:t+1],past_key_values=cache,use_cache=True)
                            mask=stack_stream(part,'masks')[:,:,2:]
                            equal+=int((mask==full_masks[:,t:t+1,2:]).sum());total+=mask.numel();del part
                    cache=out.past_key_values;previous=next_embeddings;parts.append(out.logits)
                cached=torch.cat(parts,dim=1)
                a=F.log_softmax(full.float(),dim=-1);b=F.log_softmax(cached.float(),dim=-1)
                kl=(a.exp()*(a-b)).sum(-1)
                results.append({'signal':signal,'keep':keep,'domain':meta['domain'],'sequence_id':meta['sequence_id'],
                    'tokens':ids.shape[1],'mask_agreement':equal/total if total else None,
                    'mean_kl_full_to_cached':float(kl.mean()),'max_token_kl_full_to_cached':float(kl.max()),
                    'top1_agreement':float((full.argmax(-1)==cached.argmax(-1)).float().mean()),
                    'max_abs_logit_difference':float((full-cached).abs().max())})
                del cache,full,cached,parts
        del p
    atomic_json(ROOT/'cached-decode-audit.json',{'results':results,
        'precision':'BF16 backbone; original FP32 predictor parameters with BF16 autocast',
        'scope':'Three validation documents, 16 tokens each; numerical integration check, not held-out quality estimate',
        'note':'FP32 tiny-model exact mask tests passed; pretrained BF16 differences are measured and not waived as exact equality.'})


if __name__=='__main__':
    from scripts.neuron_guessing import backbone
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    verify(*backbone())
