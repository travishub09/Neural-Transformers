"""Sequential-cached evaluation matching saved planner document records."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from pathpred.collection import atomic_json,atomic_tensors,file_sha256
from pathpred.neuro_features import capture_neuro,feature_tuple,predicted_masks,target_ids
from pathpred.neuroprefetcher import nearest_centroid
from pathpred.neuro_calibration import banks_at_count,intersection_retention,calibrate_count
from scripts.neuron_guessing import backbone
from scripts.neuro_train import load_banks,predictor,validation_data
from scripts.planner_experiment import batches
from scripts.oracle_quality import aggregate,bootstrap_quality


def infer(p,trace,banks,thresholds):
    key=f'gate_{p.config.prefix-1}'
    context=nearest_centroid(trace['activations'][(p.config.prefix-1,'gate')].abs()>=thresholds[key],banks[key])
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=p(feature_tuple(trace),context)
    return logits[0].argmax(-1),logits[1].argmax(-1)


@torch.no_grad()
def validation_predictions(model,controller,p,banks,thresholds,corpus,validation,path):
    if path.exists():return load_file(str(path),device='cuda')
    gs=[];us=[]
    for metadata,ids,lengths in batches(corpus,validation,16):
        cache=None;attention=torch.arange(ids.shape[1],device='cuda')[None,:]<lengths[:,None]
        for token in range(int(lengths.max())-1):
            valid=token+1<lengths
            with controller.use_gates(),capture_neuro(model,targets=False) as trace:
                output=model.model(ids[:,token:token+1],attention_mask=attention[:,:token+1],
                    past_key_values=cache,use_cache=True)
            cache=output.past_key_values;gate,up=infer(p,trace,banks,thresholds)
            gs.append(gate[:,0][valid].cpu());us.append(up[:,0][valid].cpu())
            if token%64==0:print(json.dumps({'stage':'validation_prefix_predictions','position':token}),flush=True)
    tensors={'gate':torch.cat(gs),'up':torch.cat(us)}
    atomic_tensors(path,tensors)
    return {k:v.cuda() for k,v in tensors.items()}


@torch.no_grad()
def evaluate(model,controller,p,original_banks,thresholds,active_banks,corpus,selected,path,calibration,*,oracle=False,dense_prefix=2):
    if path.exists():return json.loads(path.read_text())
    rows=[];started=time.monotonic()
    for metadata,ids,lengths in batches(corpus,selected,16):
        b=ids.shape[0];teacher_cache=student_cache=None;previous=None
        attention=torch.arange(ids.shape[1],device='cuda')[None,:]<lengths[:,None]
        sums=torch.zeros(b,5,device='cuda',dtype=torch.float64);kept=torch.zeros(b,device='cuda',dtype=torch.float64)
        reused=kept.clone();reuse_denominator=kept.clone();ratio_sum=kept.clone();ratio_count=kept.clone()
        centroid_correct=torch.zeros(b,2,26,device='cuda',dtype=torch.float64)
        for token in range(int(lengths.max())-1):
            valid=token+1<lengths;current=ids[:,token:token+1];mask=attention[:,:token+1]
            with controller.use_gates(),capture_neuro(model,targets=True) as trace:
                teacher=model(current,attention_mask=mask,past_key_values=teacher_cache,use_cache=True)
            teacher_cache=teacher.past_key_values
            _,centroid_targets=target_ids(trace,thresholds,active_banks if oracle else original_banks)
            gate,up=centroid_targets if oracle else infer(p,trace,original_banks,thresholds)
            centroid_correct[:,0]+=(gate[:,0]==centroid_targets[0][:,0])*valid[:,None]
            centroid_correct[:,1]+=(up[:,0]==centroid_targets[1][:,0])*valid[:,None]
            decisions=predicted_masks(gate,up,active_banks)
            decisions={layer:value for layer,value in decisions.items() if layer>=dense_prefix}
            # Prefix is dense for both teacher and student, so the same cached
            # prefix features are available causally before either sparse suffix.
            with controller.use_gates({(layer,'mlp'):value for layer,value in decisions.items()}):
                student=model(current,attention_mask=mask,past_key_values=student_cache,use_cache=True)
            student_cache=student.past_key_values
            active=torch.stack([decisions[layer][:,0] for layer in range(dense_prefix,28)],dim=1)
            kept+=active.double().mean((1,2))*valid
            if previous is not None:
                numerator=(active&previous).sum((1,2));denominator=active.sum((1,2))
                nonempty=valid&(denominator>0)
                reused+=numerator*valid;reuse_denominator+=denominator*valid
                ratio_sum+=torch.where(nonempty,numerator.double()/denominator.clamp_min(1),0.)
                ratio_count+=nonempty
            previous=active
            a=teacher.logits[:,0].float();z=student.logits[:,0].float()
            al=F.log_softmax(a,-1);zl=F.log_softmax(z,-1);target=ids[:,token+1,None]
            values=torch.stack((-al.gather(-1,target)[:,0],-zl.gather(-1,target)[:,0],
                (al.exp()*(al-zl)).sum(-1),(a.argmax(-1)==z.argmax(-1)).float()),dim=1)
            sums[:,:4]+=values.double()*valid[:,None];sums[:,4]+=valid
            if token%64==0:print(json.dumps({'stage':'cached_quality','position':token,'documents_done':len(rows)}),flush=True)
        for i,meta in enumerate(metadata):
            r=dict(zip(('teacher_nll','student_nll','kl','top1_agreement','tokens'),sums[i].tolist()))
            r.update(document=meta['document_sha256'],domain=meta['domain'],routed_kept=float(kept[i]/r['tokens']),
                reused_active=int(reused[i]),reuse_active_denominator=int(reuse_denominator[i]),
                token_reuse_ratio_sum=float(ratio_sum[i]),token_reuse_count=int(ratio_count[i]),
                centroid_correct=centroid_correct[i].cpu().tolist())
            rows.append(r)
    def summarize(rs):
        result=aggregate(rs);result.update(bootstrap_quality(rs,2000,20261113))
        result['routed_kept']=sum(r['routed_kept']*r['tokens'] for r in rs)/result['tokens']
        denominator=sum(r['reuse_active_denominator'] for r in rs);count=sum(r['token_reuse_count'] for r in rs)
        result['previous_token_reuse_count_weighted']=sum(r['reused_active'] for r in rs)/denominator if denominator else None
        result['previous_token_reuse_token_mean']=sum(r['token_reuse_ratio_sum'] for r in rs)/count if count else None
        counts=sum(torch.tensor(r['centroid_correct'],dtype=torch.float64) for r in rs)/result['tokens']
        result['gate_centroid_top1']=float(counts[0].mean());result['up_centroid_top1']=float(counts[1].mean())
        result['per_layer_centroid_top1']=counts.tolist()
        return result
    value={'overall':summarize(rows),'domains':{d:summarize([r for r in rows if r['domain']==d]) for d in ('web','code','conversation')},
        'calibration':calibration,'seconds':time.monotonic()-started,'execution':'Sequential cached teacher and student; batch16; same planner order/padding',
        'reuse_definition':'Current active rows also active in immediately previous token; first tokens excluded',
        'centroid_accuracy_labels':('True nearest labels in the calibrated runtime vocabulary; perfect by construction' if oracle else
            'Original training vocabulary, before runtime mask-density calibration; actual cached held-out execution'),
        'selection':'Privileged dense-teacher nearest-centroid labels; no predictor' if oracle else 'Trained predictor',
        'execution_dense_prefix':dense_prefix,
        'down_execution':'Full dense down projection applied to masked intermediate; no output dimensions dropped'}
    atomic_json(path.with_name(path.stem+'-documents.json'),rows);atomic_json(path,value)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/neuroprefetcher.json')
    parser.add_argument('--budget',type=float,required=True);a=parser.parse_args();c=json.loads(Path(a.config).read_text())
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(c['root'])/f'budget-{round(100*a.budget)}'
    assert json.loads((root/'training.json').read_text())['status']=='complete','Accuracy stopping rule did not pass'
    p=predictor(c);meta=json.loads((root/'checkpoint-13000000.json').read_text());weights=root/'predictor-13000000.safetensors'
    assert file_sha256(weights)==meta['file']['sha256'];p.load_state_dict(load_file(str(weights),device='cuda'));p.eval()
    banks,thresholds,aggregates=load_banks(c,a.budget);model,controller=backbone();corpus,validation,test=validation_data()
    codes=validation_predictions(model,controller,p,banks,thresholds,corpus,validation,root/'validation-predictions.safetensors')
    targets=[a.budget]+(c['additional_calibration_targets'] if a.budget==.5 else [])
    for target in targets:
        label=f'{target:.4f}';calfile=root/f'calibration-{label}.json'
        if calfile.exists():cal=json.loads(calfile.read_text())
        else:
            def measure(count):
                value=intersection_retention(codes['gate'],codes['up'],banks_at_count(aggregates,count))
                print(json.dumps({'stage':'intersection_calibration','count':count,'retention':value}),flush=True)
                return value
            cal=calibrate_count(measure,target);cal['validation_documents']=validation;atomic_json(calfile,cal)
        active=banks_at_count(aggregates,cal['count_per_centroid'])
        result=evaluate(model,controller,p,banks,thresholds,active,corpus,test,root/f'evaluation-{label}.json',cal)
        if min(result['overall']['gate_centroid_top1'],result['overall']['up_centroid_top1'])<c['far_below_accuracy']:
            atomic_json(root/'stopped-test-accuracy.json',{'status':'stopped_low_centroid_accuracy',
                'target':target,'overall':result['overall'],'threshold':c['far_below_accuracy']})
            raise SystemExit(42)


if __name__=='__main__':main()
