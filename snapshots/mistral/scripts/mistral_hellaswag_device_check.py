"""Measure dense HellaSwag numerical drift between the original and follow-up GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from pathpred.collection import atomic_json
from scripts.mistral_hellaswag_matched import ROOT, OUT, paired


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline-epoch',type=float,required=True);args=ap.parse_args()
    dest=OUT/'dense-device-check';dest.mkdir(parents=True,exist_ok=True)
    status='failed';started=time.time()
    try:
        subprocess.run([sys.executable,'-u','-m','scripts.mistral_hellaswag','--kind','dense','--output-dir',str(dest)],
            check=True,timeout=max(1,args.deadline_epoch-time.time()-90))
        old=json.loads((ROOT/'hellaswag/dense.json').read_text());new=json.loads((dest/'dense.json').read_text())
        assert old['corpus_sha256']==new['corpus_sha256'] and old['examples']==new['examples']==10042
        read=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
        a=read(dest/'dense-examples.jsonl');b=read(ROOT/'hellaswag/dense-examples.jsonl')
        aa={x['id']:x for x in a};bb={x['id']:x for x in b};assert aa.keys()==bb.keys()
        atomic_json(OUT/'dense-device-comparison.json',dict(original=old,followup=new,
            normalized_gap=paired(a,b),
            raw_prediction_disagreements=sum(aa[i]['prediction']!=bb[i]['prediction'] for i in aa),
            normalized_prediction_disagreements=sum(aa[i]['prediction_normalized']!=bb[i]['prediction_normalized'] for i in aa),
            interpretation='Full dense rerun on H200; original full baseline on A100. Matched planner results must report both dense references.'))
        status='complete'
    finally:atomic_json(dest/'completion.json',dict(status=status,started_epoch=started,ended_epoch=time.time()))

if __name__=='__main__':main()
