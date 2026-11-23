"""Finite, guarded inference-only follow-up suite."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from pathpred.collection import atomic_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-mistral',action='store_true');skip_mistral=parser.parse_args().skip_mistral
    root=Path('artifacts/neuro-controls-v1');launch=root/'launch';launch.mkdir(parents=True,exist_ok=True)
    start=time.time()
    atomic_json(launch/'progress.json',dict(pid=os.getpid(),stage='suite',status='running',started_epoch=start))
    try:
        for module,args in [('scripts.neuro_oracle_control',['--root',str(root/'oracle')]),
                            ('scripts.neuro_mistral_reference',['--root',str(root/'mistral')]),
                            ('scripts.neuro_relaxed_qwen',['--root',str(root/'relaxed-qwen')])]:
            if module=='scripts.neuro_mistral_reference' and skip_mistral:continue
            print(json.dumps({'stage':module,'status':'starting'}),flush=True)
            subprocess.run([sys.executable,'-u','-m',module,*args],check=True)
        atomic_json(launch/'completion.json',dict(status='complete',seconds=time.time()-start))
    except BaseException as e:
        atomic_json(launch/'completion.json',dict(status='failed',error=str(e),seconds=time.time()-start));raise
if __name__=='__main__':main()
