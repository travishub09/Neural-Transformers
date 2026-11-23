"""Run released NeuroPrefetcher artifacts through their published quality engine."""
import argparse
import gc
import inspect
import json
import math
from pathlib import Path
import shutil
import sys
import time
import textwrap
import types
import urllib.request
import torch
from torch.nn import functional as F
from pathpred.collection import atomic_json, file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
    reference=Path('/workspace/neuroprefetcher-reference');assets=reference/'assets'
    names=['gate_up_predictor_mistral_30p_d4_t512_k512.pt',
           'centroids_gate_512_sparsity_30.pt','centroids_up_512_sparsity_30.pt']
    for name in names:
        path=assets/name
        if not path.exists():
            print(json.dumps({'stage':'download_release','asset':name}),flush=True)
            urllib.request.urlretrieve('https://github.com/nobeldhar/NeuroPrefetcher/releases/download/v1.0-icpp2026/'+name,str(path)+'.tmp')
            Path(str(path)+'.tmp').replace(path)
    from huggingface_hub import HfApi,snapshot_download
    modeldir=Path('/workspace/pathpred/checkpoints/mistral-7b-v0.1-reference')
    if not (modeldir/'model.safetensors.index.json').exists():
        assert shutil.disk_usage('/workspace').free>16*1024**3,'Insufficient persistent space for Mistral; preserve existing artifacts'
    revision=HfApi().model_info('mistralai/Mistral-7B-v0.1').sha
    print(json.dumps({'stage':'download_model','revision':revision}),flush=True)
    snapshot_download('mistralai/Mistral-7B-v0.1',revision=revision,local_dir=str(modeldir),
        allow_patterns=['*.safetensors','*.json','tokenizer.model'],max_workers=4)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(modeldir)
    raw=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    seq=tokenizer('\n\n'.join(raw['text']),return_tensors='pt').input_ids[0,:2048].tolist()
    assert len(seq)==2048
    atomic_json(root/'protocol.json',{'reference_commit':'4c47f08b37e5b498ac0056c874b7aea45fc616fa',
        'model':'mistralai/Mistral-7B-v0.1','revision':revision,'sparsity_code':30,'dense_prefixes':[32,20],
        'sequence':seq,'assets':{n:file_sha256(assets/n) for n in names},
        'reported_ppl':{'32':3.9937,'20':5.6903},
        'note':'Released weights and official engine validation; not a newly trained Mistral planner or validation of every Qwen adaptation.'})
    sys.path.insert(0,str(reference/'quality_study'))
    from engine.decode_engine_a100 import A100DecodeEngine
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    engine=A100DecodeEngine();engine.initialize(str(modeldir),sparsity=30,dense=32,max_seq_len=2112,assets_dir=str(assets))
    original=engine.run_predictor_gpu
    stats={'active':0,'possible':0}
    def instrument(*args,**kwargs):
        result=original(*args,**kwargs)
        stats['active']+=sum(v.numel() for v in result.values());stats['possible']+=len(result)*14336
        return result
    engine.run_predictor_gpu=instrument
    dense=[];results={}
    for label,prefix in (('32',32),('20',20),('20_full_down',20)):
        if label=='20_full_down':
            # Preserve every released-engine operation except the algebraically
            # equivalent multiplication of down by an explicitly zero-filled vector.
            original_forward=engine.forward_one_token.__func__
            source=textwrap.dedent(inspect.getsource(original_forward))
            old="mlp_out = layer['mlp_down'][:, active_ids] @ a"
            assert source.count(old)==1
            source=source.replace(old,"dense_a = torch.zeros(INTERMEDIATE_SIZE, dtype=a.dtype, device=a.device)\n            dense_a[active_ids] = a\n            mlp_out = layer['mlp_down'] @ dense_a")
            namespace=dict(original_forward.__globals__);exec(source,namespace)
            engine.forward_one_token=types.MethodType(namespace['forward_one_token'],engine)
        engine.dense_layers=list(range(prefix));engine.sparse_layers=list(range(prefix,32))
        engine._sparse_layer_idx=torch.tensor([i-1 for i in engine.sparse_layers],device='cuda',dtype=torch.long) if prefix<32 else None
        engine.reset();stats.update(active=0,possible=0);nll=kl=top=0.;start=time.time()
        with torch.inference_mode():
            for pos,tid in enumerate(seq[:-1]):
                logits=engine.forward_one_token(tid,pos).float();lp=F.log_softmax(logits,-1)
                nll-=float(lp[seq[pos+1]])
                if prefix==32:dense.append(lp.cpu())
                else:
                    teacher=dense[pos].cuda();kl+=float((teacher.exp()*(teacher-lp)).sum());top+=float(teacher.argmax()==lp.argmax())
                if pos%128==0:print(json.dumps({'stage':'mistral_reference','policy':label,'prefix':prefix,'position':pos,'seconds':time.time()-start}),flush=True)
        results[label]={'nll':nll,'tokens':2047,'paper_denominator_ppl':math.exp(nll/2048),
            'correct_denominator_ppl':math.exp(nll/2047),'kl':kl/2047,'top1_agreement':top/2047 if prefix<32 else 1.,
            'suffix_kept':stats['active']/stats['possible'] if stats['possible'] else 1.,'seconds':time.time()-start}
        atomic_json(root/'results.json',results)
    engine.predictor.close()
    print(json.dumps({'stage':'mistral_done','results':results}),flush=True)


if __name__=='__main__':main()
