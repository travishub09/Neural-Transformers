"""Retokenize the fixed held-out windows; build native Mistral training text.

Reconstructing the small source corpus locally avoids a paid GPU data transfer.
Its tensor and metadata hashes must match the committed original manifest.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json,atomic_tensors,file_sha256
from scripts.oracle_quality import select_documents

MODEL='mistralai/Mistral-7B-v0.1'
REVISION='27d67f1b5f57dc0953326b2601d68371d40ea8da'


def prepare_command():
    # Streaming dataset workers may linger at interpreter shutdown on macOS.
    # main() has already atomically fsynced its complete corpus before exiting.
    return [sys.executable,'-u','-c','from scripts.prepare_corpus import main; import os,sys; main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reconstruct',action='store_true');args=parser.parse_args()
    qcfg=json.loads(Path('configs/qwen3-1.7b-base.json').read_text())
    snapshot_download(qcfg['model_id'],revision=qcfg['revision'],allow_patterns=['*token*','*.json'])
    snapshot_download(MODEL,revision=REVISION,allow_patterns=['*token*','*.json','*.model'])
    source=Path('data/stage3-corpus-v1')
    if args.reconstruct and not (source/'manifest.json').exists():
        subprocess.run(prepare_command()+['--config','configs/collection.json',
                        '--output',str(source)],check=True)
    expected=json.loads(Path('docs/step3/corpus-manifest.json').read_text())
    for item in expected['files']:
        assert file_sha256(source/item['file'])==item['sha256'],f'Original corpus reconstruction mismatch: {item["file"]}'
    qtok=AutoTokenizer.from_pretrained(qcfg['model_id'],revision=qcfg['revision'],local_files_only=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,local_files_only=True)
    ids=load_file(str(source/'tokens.safetensors'))['input_ids']
    records=[json.loads(x) for x in (source/'sequences.jsonl').read_text().splitlines()]
    chosen=select_documents(records,'validation',8,{'seed':20261111,'minimum_sequence_length':128})
    chosen+=select_documents(records,'test',24,{'seed':20261109,'minimum_sequence_length':128})
    assert sum(r['split']=='test' for r in chosen)==63
    output=Path('data/mistral-heldout-v1');output.mkdir(parents=True,exist_ok=True)
    encoded=[];metadata=[]
    for row in chosen:
        old=ids[row['sequence_id'],:row['length']].tolist()
        # EOS in the original tokenizer is a boundary, never literal training text.
        text=qtok.decode(old,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        value=tok(text,add_special_tokens=False)['input_ids']
        if old[-1]==qtok.eos_token_id:value.append(tok.eos_token_id)
        assert len(value)>1
        new=dict(row,sequence_id=len(metadata),length=len(value),source_sequence_id=row['sequence_id'],
                 source_window_tokens=len(old),text_sha256=__import__('hashlib').sha256(text.encode()).hexdigest())
        metadata.append(new);encoded.append(value)
    maximum=max(map(len,encoded));padded=[v+[tok.eos_token_id]*(maximum-len(v)) for v in encoded]
    tensors=atomic_tensors(output/'tokens.safetensors',{'input_ids':torch.tensor(padded,dtype=torch.int32)})
    (output/'sequences.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata))
    atomic_json(output/'manifest.json',{'model_id':MODEL,'revision':REVISION,'source_files':expected['files'],
        'documents':len(metadata),'max_length':maximum,'files':[tensors,{'file':'sequences.jsonl','sha256':file_sha256(output/'sequences.jsonl')}],
        'protocol':'Same 63 test and 24 validation document windows, decoded with source tokenizer and retokenized with Mistral. All Mistral methods score identical resulting text; no Qwen token IDs reused.'})
    modelcfg={'model_id':MODEL,'revision':REVISION}
    atomic_json(Path('configs/mistral-7b-base.json'),modelcfg)
    output=Path('data/mistral-pilot-train-v1')
    if not (output/'manifest.json').exists():
        subprocess.run(prepare_command()+['--config','configs/mistral-pilot-corpus.json',
            '--model-config','configs/mistral-7b-base.json','--output',str(output)],check=True)
    print(json.dumps({'status':'complete','heldout_documents':len(metadata),'train':str(output)}),flush=True)


if __name__=='__main__':main()
