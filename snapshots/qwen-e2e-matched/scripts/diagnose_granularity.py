"""Measure neuron/group concentration from original cached activations."""
import argparse
import json
from pathlib import Path
import time
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from pathpred.collection import atomic_json, atomic_tensors, file_sha256
from pathpred.oracle_checks import concentration


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('artifacts/granularity-v1'))
    a=p.parse_args()
    torch.set_num_threads(4)
    source=Path('data/stage3-full-v1');clusters=Path('data/stage4-clusters-v1')
    cm=json.loads((clusters/'manifest.json').read_text())
    collection=json.loads((source/'manifest.json').read_text())
    fp={str(f):file_sha256(Path(f)) for f in ['scripts/diagnose_granularity.py','pathpred/oracle_checks.py',
        'configs/phase0-followup.json','docs/phase0-followup-protocol.md',str(source/'manifest.json'),str(clusters/'manifest.json')]}
    manifest_path=a.output/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text());assert old['fingerprint']==fp
        if old['status']=='complete':
            for entry in old['files'].values():assert file_sha256(a.output/entry['file'])==entry['sha256']
            print('Granularity diagnostic already complete',flush=True);return
    a.output.mkdir(parents=True,exist_ok=True)
    atomic_json(manifest_path,{'status':'in_progress','fingerprint':fp})
    per_layer=[]
    for layer in cm['layers']:
        entry=layer['tensors'];assert file_sha256(clusters/entry['file'])==entry['sha256']
        per_layer.append(load_file(str(clusters/entry['file'])))
    learned=torch.stack([x['permutation'] for x in per_layer]).cuda()
    random=torch.stack([x['random_permutation'] for x in per_layer]).cuda()
    recorded_norms=torch.stack([x['sample_group_norms'] for x in per_layer],1)
    with safe_open('data/stage6-phase0-v1/examples.safetensors',framework='pt') as f:
        index={k:f.get_tensor(k) for k in ['sequence_ids','token_positions','split_codes','documents','domains']}
    data_manifest=json.loads(Path('data/stage6-phase0-v1/manifest.json').read_text())
    pieces={};offset=0;start=time.monotonic();maximum_norm_error=0.
    for shard_number,shard in enumerate(collection['shards']):
        path=source/shard['file'];assert file_sha256(path)==shard['sha256']
        with safe_open(str(path),framework='pt') as f:
            raw=f.get_tensor('sample_mlp');n=len(raw)
            assert torch.equal(f.get_tensor('sample_sequence_ids'),index['sequence_ids'][offset:offset+n])
            assert torch.equal(f.get_tensor('sample_token_positions'),index['token_positions'][offset:offset+n])
        if not n:continue
        x=raw.cuda().float();energy=x.square()
        reordered=energy.gather(-1,learned[None].expand(n,-1,-1))
        randomized=energy.gather(-1,random[None].expand(n,-1,-1))
        views={'neurons':energy,
               'original128':energy.reshape(n,28,48,128).sum(-1),
               'learned128':reordered.reshape(n,28,48,128).sum(-1),
               'random128':randomized.reshape(n,28,48,128).sum(-1),
               'learned_subgroups32':reordered.reshape(n,28,192,32).sum(-1)}
        expected=recorded_norms[offset:offset+n].cuda()
        torch.testing.assert_close(views['learned128'].sqrt(),expected,rtol=1e-5,atol=1e-5)
        maximum_norm_error=max(maximum_norm_error,float((views['learned128'].sqrt()-expected).abs().max()))
        total=energy.sum(-1)
        for name,values in views.items():
            torch.testing.assert_close(values.sum(-1),total,rtol=1e-5,atol=1e-5)
            for metric,value in concentration(values).items():
                pieces.setdefault(name+'__'+metric,[]).append(value.cpu())
        rms=energy.mean(-1).sqrt()
        pieces.setdefault('neurons__exact_zero_fraction',[]).append((x==0).float().mean(-1).cpu())
        for epsilon in [.001,.01,.1]:
            near=(x.abs()<=epsilon*rms[...,None]).float().mean(-1)
            pieces.setdefault('neurons__fraction_at_most_'+str(epsilon)+'_rms',[]).append(near.cpu())
        offset+=n
        if shard_number%10==0:
            print(json.dumps({'checked_shards':shard_number+1,'sample_positions':offset,'seconds':time.monotonic()-start}),flush=True)
        del raw,x,energy,reordered,randomized,views,expected
    assert offset==len(index['sequence_ids'])
    statistics={k:torch.cat(v) for k,v in pieces.items()}
    output={'overall':{},'by_split':{},'by_domain':{}}
    def summarize(mask):
        return {'positions':int(mask.sum()),'metrics':{k:{'mean':float(v[mask].mean()),'by_layer':v[mask].mean(0).tolist()}
                                                      for k,v in statistics.items()}}
    output['overall']=summarize(torch.ones(offset,dtype=torch.bool))
    for name,code in [('train',0),('validation',1),('test',2)]:
        output['by_split'][name]=summarize(index['split_codes']==code)
    for code,name in enumerate(data_manifest['domains']):
        output['by_domain'][name]=summarize((index['domains']==code)&(index['split_codes']==2))
    files={'statistics':atomic_tensors(a.output/'statistics.safetensors',{**statistics,**index})}
    report={'status':'complete','fingerprint':fp,'results':output,'positions':offset,'seconds':time.monotonic()-start,
        'checks':{'group_energy_additive':True,'all_group_norms_reproduce_step4':True,'maximum_group_norm_abs_error':maximum_norm_error},
        'notes':['Squared activation energy is additive across neurons/groups; raw norm sums are not',
                 '32-neuron groups subdivide learned128 groups; no new 32-group clustering fitted',
                 'Concentration and numerical zeros do not establish causal importance or safe skipping']}
    atomic_json(a.output/'results.json',report)
    atomic_json(manifest_path,{'status':'complete','fingerprint':fp,'files':files})
    print(json.dumps({'status':'complete','positions':offset,'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
