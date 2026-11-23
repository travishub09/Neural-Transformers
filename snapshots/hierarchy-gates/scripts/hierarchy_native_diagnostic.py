"""Evaluate the trained D50 policy at its native zero shift, without updates."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from pathpred.collection import atomic_json,file_sha256
from pathpred.hierarchical import DecodeCost
from scripts import hierarchy_experiment as h
from scripts.neuron_guessing import backbone,CAL
from scripts.calibrate_native import selections


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    a=parser.parse_args();c=h.protocol(a.config);root=Path(c['root'])/'D/budget-50'
    output=root/'native-final.json'
    if output.exists():return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,controller=backbone();p,offsets,_=h.warm(50,'D')
    meta=json.loads((root/'checkpoint-02000000.json').read_text())
    path=root/'predictor-02000000.safetensors';assert file_sha256(path)==meta['file']['sha256']
    assert meta['fingerprint']==h.fingerprint(a.config,50)
    p.load_state_dict(load_file(str(path),device='cuda'))
    offsets=torch.tensor(meta['base_offsets'],device='cuda')
    q=load_file(str(CAL),device='cuda')['q75'];corpus,_,test=selections()
    result=h.evaluate(model,controller,p,q,offsets,0.,DecodeCost.from_model(model,p,c['reference_context']),corpus,test,output)
    result.update(evaluation_calibration='native_zero_shift_unmatched',matched_budget=False,
                  diagnostic='No training updates; same final D50 checkpoint, original native offsets, no global calibration shift.',
                  checkpoint_sha256=meta['file']['sha256'])
    atomic_json(output,result)


if __name__=='__main__':main()
