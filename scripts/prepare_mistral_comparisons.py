"""Pinned released baseline assets and standard HellaSwag validation examples."""
import json,re,subprocess,urllib.request
from pathlib import Path
from pathpred.collection import atomic_json,file_sha256
REFERENCE_COMMIT='4c47f08b37e5b498ac0056c874b7aea45fc616fa'
ROOT=Path('artifacts/mistral-comparisons-v1')

def preprocess(text):
    return re.sub(r'\[.*?\]','',text.strip().replace(' [title]','. ')).replace('  ',' ')

def process(doc):
    return dict(query=preprocess(doc['activity_label']+': '+doc['ctx_a']+' '+doc['ctx_b'].capitalize()),
        choices=[preprocess(x) for x in doc['endings']],gold=int(doc['label']))

def encode_pair(tok,context,continuation):
    # lm-evaluation-harness shifts trailing context spaces into the continuation.
    spaces=len(context)-len(context.rstrip())
    if spaces:continuation=context[-spaces:]+continuation;context=context[:-spaces]
    # Explicit BOS, identical for dense and sparse Mistral. Record this task option.
    a=tok.encode(context,add_special_tokens=True)
    full=tok.encode(context+continuation,add_special_tokens=True)
    assert full[:len(a)]==a,'Tokenizer boundary changed; refuse incorrect continuation scoring'
    assert len(full)>len(a)
    return full,len(a)

def main():
    ROOT.mkdir(parents=True,exist_ok=True);ref=Path('/workspace/neuroprefetcher-reference')
    if not (ref/'.git').exists():subprocess.run(['git','clone','https://github.com/nobeldhar/NeuroPrefetcher',str(ref)],check=True)
    subprocess.run(['git','-C',str(ref),'checkout','--detach',REFERENCE_COMMIT],check=True)
    assets=ref/'assets';assets.mkdir(exist_ok=True)
    names=['gate_up_predictor_mistral_30p_d4_t512_k512.pt','centroids_gate_512_sparsity_30.pt','centroids_up_512_sparsity_30.pt']
    expected=json.loads((ROOT/'wikitext-reference.json').read_text())['assets']
    for name in names:
        path=assets/name
        if not path.exists():
            print(json.dumps(dict(stage='baseline_asset',name=name)),flush=True)
            urllib.request.urlretrieve('https://github.com/nobeldhar/NeuroPrefetcher/releases/download/v1.0-icpp2026/'+name,str(path)+'.tmp')
            Path(str(path)+'.tmp').replace(path)
        assert file_sha256(path)==expected[name],name
    atomic_json(ROOT/'baseline-assets.json',dict(reference_commit=REFERENCE_COMMIT,assets=expected))
    dest=Path('data/hellaswag-mistral-v1');dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():return
    from huggingface_hub import HfApi
    from datasets import load_dataset
    from transformers import AutoTokenizer
    rev=HfApi().dataset_info('Rowan/hellaswag').sha
    ds=load_dataset('Rowan/hellaswag',revision=rev,split='validation')
    tok=AutoTokenizer.from_pretrained('checkpoints/mistral-7b-v0.1',local_files_only=True)
    tmp=dest/'examples.tmp'
    with tmp.open('w') as f:
        for i,doc in enumerate(ds):
            row=process(doc);row['id']=i;row['sequences']=[];row['prompt_lengths']=[]
            for ending in row['choices']:
                ids,n=encode_pair(tok,row['query'],' '+ending)
                row['sequences'].append(ids);row['prompt_lengths'].append(n)
            assert len(set(row['prompt_lengths']))==1
            f.write(json.dumps(row)+'\n')
    tmp.replace(dest/'examples.jsonl')
    assert len(ds)==10042
    atomic_json(dest/'manifest.json',dict(dataset='Rowan/hellaswag',revision=rev,split='validation',examples=len(ds),
        zero_shot=True,add_bos_token=True,acc_norm_denominator='ending characters, excluding added target delimiter',
        prompt_execution='fully dense; candidate histories reset independently',
        source='EleutherAI/lm-evaluation-harness/lm_eval/tasks/hellaswag',
        task_reference_commit='d6de81643928d653435c431bae19945d41d32520',
        file_sha256=file_sha256(dest/'examples.jsonl')))
if __name__=='__main__':main()
