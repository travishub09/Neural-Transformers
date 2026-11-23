"""Guard-compatible standalone reference check, without private training data."""
import json,os,sys,time
from pathlib import Path
from pathpred.collection import atomic_json
from scripts.neuro_mistral_reference import main as run

def main():
    root=Path('artifacts/neuro-mistral-v1');launch=root/'launch';launch.mkdir(parents=True,exist_ok=True);start=time.time()
    atomic_json(launch/'progress.json',dict(pid=os.getpid(),stage='suite',status='running',started_epoch=start))
    try:
        sys.argv=[sys.argv[0],'--root',str(root)];run()
        atomic_json(launch/'completion.json',dict(status='complete',seconds=time.time()-start))
    except BaseException as e:
        atomic_json(launch/'completion.json',dict(status='failed',error=str(e),seconds=time.time()-start));raise
if __name__=='__main__':main()
