"""Give the hidden predictor the same 3M E2E pilot exposure as the binary model."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json,file_sha256
from pathpred.neuron_prediction import NeuronPredictor,neuron_config
from scripts import e2e_pilot as pilot


def main():
    pilot.ROOT=Path('artifacts/e2e-hidden-pilot-v1');pilot.ROOT.mkdir(parents=True,exist_ok=True)
    warm=Path('artifacts/neuron-guessing-v1/hidden.safetensors')
    cfg=json.loads(Path('configs/e2e-pilot.json').read_text())
    cfg['warm_start_sha256']=file_sha256(warm)
    cfg['scope']='Hidden-state comparator; exactly the original 2M@50% + 1M@40% pilot schedule and training windows.'
    pilot.config=lambda:cfg
    atomic_json(pilot.ROOT/'config.json',cfg)
    def predictor():
        p=NeuronPredictor(neuron_config('hidden')).cuda().eval()
        p.load_state_dict(load_file(str(warm),device='cuda'));return p
    pilot.predictor=predictor
    base_fingerprint=pilot.source_fingerprint
    def fingerprint():
        fp=base_fingerprint()
        for f in [Path(__file__),pilot.ROOT/'config.json',warm]:fp[str(f)]=file_sha256(f)
        return fp
    pilot.source_fingerprint=fingerprint
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:pilot.run()
    except BaseException as e:
        atomic_json(pilot.ROOT/'completion.json',{'status':'failed','error_type':type(e).__name__,'error':str(e)});raise
    else:atomic_json(pilot.ROOT/'completion.json',{'status':'complete'})


if __name__=='__main__':main()
