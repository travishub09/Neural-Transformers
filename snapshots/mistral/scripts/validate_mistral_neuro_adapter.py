"""Validate the released predictor adapter against its official A100 quality engine."""
import json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from pathpred.collection import atomic_json
from scripts.mistral_comparisons import load,ROOT
from scripts.mistral_validation_cache import fingerprint,reuse_or_archive

@torch.no_grad()
def main():
    dest=ROOT/'neuro-adapter-validation.json'
    reference=Path('/workspace/neuroprefetcher-reference/quality_study')
    fp=fingerprint([Path(__file__),Path('scripts/mistral_validation_cache.py'),
        Path('scripts/mistral_comparison_core.py'),Path('scripts/mistral_comparisons.py'),
        Path('scripts/mistral_planner.py'),Path('pathpred/gated_qwen.py'),
        Path('configs/mistral-planner-phased.json'),ROOT/'baseline-assets.json',ROOT/'wikitext-reference.json',
        reference/'engine/decode_engine_a100.py',reference/'np_python_v2/inproc_predictor.py',
        reference/'np_python_v2/centroid_bank.py'])
    if reuse_or_archive(dest,fp):return
    sys.path.insert(0,'/workspace/neuroprefetcher-reference/quality_study')
    from engine.decode_engine_a100 import A100DecodeEngine,head_attn_sketch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    policy=load('neuro',50,20)
    engine=A100DecodeEngine();engine.initialize('checkpoints/mistral-7b-v0.1',sparsity=30,dense=20,
        max_seq_len=256,assets_dir='/workspace/neuroprefetcher-reference/assets')
    # Exact predictor feature replay checks centroid head indexing, bank intersection and cap.
    original=engine.run_predictor_gpu;matches=[];feature_log=[]
    def compare(E,P,H,N,g0):
        result=original(E,P,H,N,g0)
        net=policy.neuro.net
        gl,ul=net(E[None].half(),P[None].half(),H[None].half(),torch.tensor([[N]],device='cuda',dtype=torch.float16),torch.tensor([g0],device='cuda'))
        masks=policy.neuro.gate[gl.argmax(-1)]&policy.neuro.up[ul.argmax(-1)]
        masks=masks&(masks.cumsum(-1)<=policy.neuro.cap)
        ok=all(torch.equal(masks[0,l-1].nonzero().flatten(),v) for l,v in result.items())
        matches.append(ok);assert ok,'Released feature replay changed active neuron IDs'
        return result
    engine.run_predictor_gpu=compare
    seq=json.loads((ROOT/'wikitext-reference.json').read_text())['sequence'][:128]
    results={}
    for prefix in (32,20,2):
        engine.dense_layers=list(range(prefix));engine.sparse_layers=list(range(prefix,32))
        engine._sparse_layer_idx=torch.tensor([l-1 for l in engine.sparse_layers],device='cuda',dtype=torch.long) if prefix<32 else None
        engine.reset();policy.reset();policy.prefix=prefix
        nlla=nllb=kl=top=0.
        for t,tid in enumerate(seq[:-1]):
            reference=F.log_softmax(engine.forward_one_token(tid,t).float(),-1)
            valid=torch.ones(1,t+1,device='cuda',dtype=torch.bool)
            logits=policy.step(torch.tensor([[tid]],device='cuda'),valid)[0]
            actual=F.log_softmax(logits,-1)
            nlla-=float(reference[seq[t+1]]);nllb-=float(actual[seq[t+1]])
            kl+=float((reference.exp()*(reference-actual)).sum());top+=float(reference.argmax()==actual.argmax())
            if t%16==0:print(json.dumps(dict(stage='adapter_validation',prefix=prefix,token=t)),flush=True)
        n=len(seq)-1
        results[str(prefix)]=dict(official_ppl=__import__('math').exp(nlla/n),adapter_ppl=__import__('math').exp(nllb/n),
            relative_ppl_difference=__import__('math').expm1((nllb-nlla)/n),kl=kl/n,top1_agreement=top/n)
    engine.predictor.close()
    # Independently reject a materially different execution before running expensive comparisons.
    passed=all(matches) and results['32']['kl']<.002 and all(abs(v['relative_ppl_difference'])<.03 and v['kl']<.03 for v in results.values())
    atomic_json(dest,dict(passed=passed,fingerprint=fp,results=results,feature_replays=len(matches),feature_replays_exact=all(matches),
        criteria=dict(dense_kl_below=.002,all_relative_ppl_gap_below=.03,all_kl_below=.03),
        note='128-token numerical smoke test against released engine; published full WikiText reproduction is separately preserved'))
    assert passed,'Adapter differs materially from released engine: do not claim a baseline comparison'
if __name__=='__main__':main()
