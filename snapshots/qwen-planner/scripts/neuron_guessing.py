"""Prepare, profile, train, then evaluate the frozen-Qwen per-neuron guessing test."""
import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from pathpred import gate_qwen3
from pathpred.collection import atomic_json,file_sha256,atomic_tensors
from pathpred.neuron_prediction import (neuron_config,NeuronPredictor,collect_dense,
    route_neurons,stack_stream,recall_weighted_loss,recovered_advantage)
from pathpred.sparsity_diagnostics import select_mask
from pathpred.reordering import load_clustered_qwen
from pathpred.oracle_checks import quality_rows
from scripts.oracle_quality import select_documents,aggregate,bootstrap_quality
from scripts.cluster_neurons import read_samples

ROOT=Path('artifacts/neuron-guessing-v1')
CAL=ROOT/'calibration.safetensors'
SIGNALS=('binary','hidden')


def cfg():
    return json.loads(Path('configs/neuron-prediction.json').read_text())


def fingerprint():
    files=['configs/neuron-prediction.json','docs/neuron-guessing-protocol.md','pathpred/neuron_prediction.py',
        'scripts/neuron_guessing.py','pathpred/path_model.py','pathpred/gated_qwen.py',
        'pathpred/reordering.py','pathpred/oracle_checks.py','scripts/oracle_quality.py',
        'data/neuron-training-corpus-v1/manifest.json','data/stage3-corpus-v1/manifest.json']
    return {f:file_sha256(Path(f)) for f in files}


def data(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['files']:
        assert file_sha256(root/item['file'])==item['sha256']
    records=[json.loads(x) for x in (root/'sequences.jsonl').read_text().splitlines()]
    return load_file(str(root/'tokens.safetensors')),records


def backbone():
    model=load_clustered_qwen('checkpoints/qwen3-1.7b-base-clustered-v1',dtype=torch.bfloat16,
        device_map={'':'cuda'},attn_implementation='eager',local_files_only=True).eval()
    controller=gate_qwen3(model,1)
    assert all(not p.requires_grad for p in model.parameters())
    return model,controller


def predictors():
    torch.manual_seed(cfg()['seed'])
    result={s:NeuronPredictor(neuron_config(s)).cuda() for s in SIGNALS}
    cal=load_file(str(CAL),device='cuda')
    with torch.no_grad():
        # Warm-start logits from train-only marginal label frequencies, not test data.
        prior=(cal['frequency_50']+cal['frequency_40'])/2
        prior=prior[2:].clamp(.001,.999)
        bias=(cfg()['positive_weight']*prior/(1-prior)).log()
        for p in result.values():
            p.output_bias.copy_(bias);p.output_weight.mul_(.01)
    return result


def prepare():
    if CAL.exists():
        print('Calibration already exists',flush=True);return
    started=time.monotonic()
    raw,seqs,pos,splits,manifest=read_samples(Path('data/stage3-corpus-v1'),Path('data/stage3-full-v1'))
    train=torch.tensor([s=='train' for s in splits]);tensors={};qs=[];freq={50:[],40:[]};teal={50:[],40:[]}
    for l in range(28):
        permutation=load_file(f'data/stage4-clusters-v1/layer-{l:02}.safetensors')['permutation']
        x=raw[l,train].cuda().float().abs()[:,permutation.cuda()]
        qs.append(x.quantile(.75,dim=0).cpu())
        for pct in [50,40]:
            freq[pct].append(select_mask(x,pct/100).float().mean(0).cpu())
            # Fixed per-layer threshold, sampled from training activations only.
            gen=torch.Generator(device='cuda').manual_seed(20261109+l)
            indices=torch.randint(x.numel(),(262144,),generator=gen,device='cuda')
            teal[pct].append(x.flatten()[indices].quantile(1-pct/100).cpu())
        print(json.dumps({'calibration_layer':l,'seconds':time.monotonic()-started}),flush=True)
    tensors['q75']=torch.stack(qs)
    for pct in [50,40]:
        tensors[f'frequency_{pct}']=torch.stack(freq[pct]);tensors[f'teal_{pct}']=torch.stack(teal[pct])
    entry=atomic_tensors(CAL,tensors)
    atomic_json(ROOT/'calibration.json',{'file':entry,'train_positions':int(train.sum()),
        'thresholds':'strict above own q75 of absolute SwiGLU intermediates; existing train split only',
        'coordinate_order':'canonical grouped checkpoint stored neuron order',
        'seconds':time.monotonic()-started})


def training_order(records):
    indices=[i for i,r in enumerate(records) if r['split']=='train' and r['length']>=2]
    g=torch.Generator().manual_seed(cfg()['seed'])
    return [indices[i] for i in torch.randperm(len(indices),generator=g).tolist()]


def one_batch(model,controller,ps,optimizers,ids,cal,keep,hide,lr,update=True):
    with torch.no_grad(),controller.use_gates(),collect_dense(model,budgets=(.5,.4),capture_hidden=False) as teacher:
        model.model(ids,use_cache=False)
    labels={k:torch.stack([teacher['labels'][l][str(k)] for l in range(2,28)],dim=2) for k in [.5,.4]}
    stats={}
    for signal,p in ps.items():
        p.eval()
        with torch.no_grad(),controller.use_gates(),route_neurons(model,controller,cal['q75'],keep,
             predictor=p,hide_probability=hide) as history:
            model.model(ids,use_cache=False)
        inputs={'states':stack_stream(history,'bits')} if signal=='binary' else {'hidden':stack_stream(history,'hidden')}
        # The rollout tensors are detached; only the predictor receives gradients.
        del history
        p.train();optimizer=optimizers[signal]
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=p.forward_tokens(**inputs)
            loss=recall_weighted_loss(logits,labels[keep],cfg()['positive_weight'])
        assert torch.isfinite(loss)
        if update:
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(p.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        else:grad=0.
        with torch.no_grad():
            chosen=select_mask(logits,keep)
            recall=float((chosen & labels[keep]).sum()/labels[keep].sum())
        stats[signal]={'loss':float(loss.detach()),'recall':recall,'gradient_norm':float(grad)}
        del logits,inputs,loss
    counts={k:torch.stack([teacher['labels'][l][str(k)].sum((0,1)) for l in range(28)]) for k in [.5,.4]}
    return stats,counts


def snapshot(path,ps,opts,position,tokens,frequencies,frequency_tokens,elapsed,fp):
    # Atomic, recoverable checkpoint. Qwen weights never enter the optimizer/checkpoint.
    payload={'models':{s:p.state_dict() for s,p in ps.items()},
        'optimizers':{s:o.state_dict() for s,o in opts.items()},'next_position':position,
        'tokens':tokens,'frequencies':frequencies,'frequency_tokens':frequency_tokens,
        'elapsed_seconds':elapsed,'fingerprint':fp,
        'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def train(pilot=False):
    c=cfg();fp=fingerprint();cal=load_file(str(CAL),device='cuda')
    corpus,records=data('data/neuron-training-corpus-v1');order=training_order(records)
    _,heldout=data('data/stage3-corpus-v1')
    forbidden={r['document_sha256'] for r in heldout if r['split']!='train'}
    assert not ({records[i]['document_sha256'] for i in order}&forbidden)
    assert sum(records[i]['length'] for i in order)>=c['train_tokens_per_predictor']
    model,controller=backbone();ps=predictors()
    opts={s:torch.optim.AdamW(p.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay']) for s,p in ps.items()}
    counts={k:torch.zeros(28,6144,dtype=torch.int64,device='cuda') for k in [.5,.4]}
    pos=seen=frequency_tokens=0;elapsed=0.;last=ROOT/'last.pt'
    if not pilot and last.exists():
        saved=torch.load(last,map_location='cuda',weights_only=True);assert saved['fingerprint']==fp
        for s in SIGNALS:ps[s].load_state_dict(saved['models'][s]);opts[s].load_state_dict(saved['optimizers'][s])
        pos=saved['next_position'];seen=saved['tokens'];counts=saved['frequencies'];frequency_tokens=saved['frequency_tokens'];elapsed=saved['elapsed_seconds']
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());del saved
    accounting={s:p.accounting() for s,p in ps.items()}
    base_params=sum(p.numel() for p in model.parameters())
    for s,a in accounting.items():
        a['dense_bf16_predictor_weight_bytes']=a['parameters']*2
        a['fp32_training_weight_bytes']=a['parameters']*4
        a['relative_to_qwen_bf16_weight_payload']=a['parameters']/base_params
        a['dense_qwen_1k_reference_flops']=3675783168
        a['dense_projection_flop_fraction_1k']=a['matmul_flops_per_llm_token']/3675783168
    atomic_json(ROOT/'accounting.json',{'predictors':accounting,'qwen_parameters':base_params,
        'packed_one_bit_history_bytes_per_token':28*6144//8,
        'actual_int8_history_bytes_per_token':28*6144,
        'notes':['Weight bytes are theoretical payload, not measured DRAM traffic.',
                 'Dense projections used here; previous-token embeddings reused.',
                 'Train-time FP32 parameters and optimizer; inference runs BF16 autocast.',
                 'Attention dense; first two MLP layers dense. Retention budgets exclude predictor overhead.']})
    start=time.monotonic();last_save=seen;timings=[];sample_tokens=0
    for j in range(pos,len(order)):
        if seen>=c['train_tokens_per_predictor']:break
        meta=records[order[j]];length=min(meta['length'],c['sequence_length'],c['train_tokens_per_predictor']-seen)
        if length<1:break
        ids=corpus['input_ids'][meta['sequence_id']:meta['sequence_id']+1,:length].long().cuda()
        keep=c['budgets'][j%2]
        # Pilot exercises fully sparse deployment conditions, including backward.
        hide=1. if pilot else min(1.,seen/c['observation_curriculum_tokens'])
        factor=min(1.,(seen+length)/c['warmup_tokens'])*(.1+.9*.5*(1+math.cos(math.pi*seen/c['train_tokens_per_predictor'])))
        torch.cuda.synchronize();tick=time.monotonic()
        stats,newcounts=one_batch(model,controller,ps,opts,ids,cal,keep,hide,c['learning_rate']*factor)
        torch.cuda.synchronize();seconds=time.monotonic()-tick;timings.append(seconds)
        sample_tokens+=length;seen+=length;frequency_tokens+=length
        for k in counts:counts[k]+=newcounts[k]
        total_elapsed=elapsed+time.monotonic()-start
        if j%20==0 or pilot:
            status={'position':j+1,'tokens_per_predictor':seen,'keep':keep,'hide_probability':hide,
                'batch_seconds':seconds,'elapsed_seconds':total_elapsed,'signals':stats,
                'tokens_per_second_joint':sample_tokens/sum(timings)}
            atomic_json(ROOT/('pilot-progress.json' if pilot else 'progress.json'),status)
            print(json.dumps(status),flush=True)
        if pilot and len(timings)>=8:
            throughput=sample_tokens/sum(timings)
            atomic_json(ROOT/'pilot.json',{'status':'complete','batches':len(timings),
                'tokens':sample_tokens,'seconds':sum(timings),'joint_tokens_per_second':throughput,
                'estimated_training_hours':c['train_tokens_per_predictor']/throughput/3600,
                'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                'accounting':accounting,'note':'Includes two online sparse rollouts, teacher labels, and both predictor updates.'})
            return
        if not pilot and (seen-last_save>=c['checkpoint_every_tokens'] or seen>=c['train_tokens_per_predictor'] or total_elapsed>c['maximum_runtime_hours']*3600):
            snapshot(last,ps,opts,j+1,seen,counts,frequency_tokens,total_elapsed,fp);last_save=seen
            if total_elapsed>c['maximum_runtime_hours']*3600:
                raise TimeoutError('Four-hour training guard; checkpoint saved, evaluation not marked complete')
    assert not pilot and seen>=c['train_tokens_per_predictor']
    for s,p in ps.items():
        atomic_tensors(ROOT/f'{s}.safetensors',{k:v.detach().cpu().contiguous() for k,v in p.state_dict().items()})
    atomic_tensors(ROOT/'fixed-frequencies.safetensors',{str(k):v.float().cpu()/frequency_tokens for k,v in counts.items()})
    atomic_json(ROOT/'training.json',{'status':'complete','tokens_per_predictor':seen,
        'distinct_windows':j+1,'seconds':elapsed+time.monotonic()-start,'fingerprint':fp,
        'config':c,'backbone_frozen':all(not p.requires_grad and p.grad is None for p in model.parameters()),
        'test_documents_disjoint':True})


def evaluate():
    c=cfg();training=json.loads((ROOT/'training.json').read_text())
    assert training['status']=='complete' and training['tokens_per_predictor']>=c['train_tokens_per_predictor']
    cal=load_file(str(CAL),device='cuda');frequencies=load_file(str(ROOT/'fixed-frequencies.safetensors'),device='cuda')
    corpus,records=data('data/stage3-corpus-v1')
    selected=select_documents(records,'test',c['test_documents_per_domain'],c)
    model,controller=backbone();ps=predictors()
    for s,p in ps.items():p.load_state_dict(load_file(str(ROOT/f'{s}.safetensors'),device='cuda'));p.eval()
    methods=['binary','hidden','fixed','oracle','dense_oracle','previous_oracle','teal_mlp','wina_mlp']
    documents=[];start=time.monotonic()
    checkpoint_hashes={s:file_sha256(ROOT/f'{s}.safetensors') for s in SIGNALS}
    for index,meta in enumerate(selected):
        length=min(meta['length'],c['sequence_length']);ids=corpus['input_ids'][meta['sequence_id']:meta['sequence_id']+1,:length].long().cuda()
        with torch.no_grad(),controller.use_gates(),collect_dense(model,budgets=c['budgets'],capture_hidden=False) as dense:
            teacher=model(ids,use_cache=False).logits
        if index==0:
            with torch.no_grad(),controller.use_gates(),route_neurons(model,controller,cal['q75'],1.,method='oracle'):
                assert torch.equal(teacher,model(ids,use_cache=False).logits)
        document={'metadata':meta,'conditions':{}}
        for keep in c['budgets']:
            fixed=select_mask(frequencies[str(keep)],keep)
            labels={l:dense['labels'][l][str(keep)] for l in range(28)}
            for method in methods:
                torch.cuda.synchronize();tick=time.monotonic()
                with torch.no_grad(),controller.use_gates(),route_neurons(model,controller,cal['q75'],keep,
                    predictor=ps.get(method),method=method,fixed=fixed,teacher_masks=labels,
                    teal_thresholds=cal[f'teal_{round(100*keep)}']) as history:
                    student=model(ids,use_cache=False).logits
                torch.cuda.synchronize();duration=time.monotonic()-tick
                metrics=quality_rows(teacher,student,ids)
                row={k:float(v.double().sum()) for k,v in metrics.items()}
                mask=stack_stream(history,'masks');bits=stack_stream(history,'bits')
                assert mask[:,:,:2].all() and (bits[~mask]==-1).all()
                if method!='teal_mlp':assert (mask[:,:,2:].sum(-1)==math.ceil(6144*keep)).all()
                targets=torch.stack([labels[l] for l in range(2,28)],2)
                row.update(tokens=length-1,domain=meta['domain'],document=meta['document_sha256'],
                    attention_retained=1.,mlp_retained=float(mask.float().mean()),
                    routed_mlp_retained=float(mask[:,:,2:].float().mean()),
                    label_recall=float((mask[:,:,2:]&targets).sum()/targets.sum()),
                    seconds=duration,
                    ideal_binary_input_row_bytes_per_token=float(((bits[:,:,:26]>=0).sum()+(bits[:,:,:26]==1).sum())/length)*256*2)
                document['conditions'][f'{method}-{round(keep*100)}']=row
                del student,history,mask,bits,targets
        atomic_json(ROOT/f'eval-window-{meta["sequence_id"]}.json',document);documents.append(document)
        del teacher,dense
        print(json.dumps({'evaluated_documents':index+1,'total':len(selected),'seconds':time.monotonic()-start}),flush=True)
    results={}
    for name in documents[0]['conditions']:
        rows=[d['conditions'][name] for d in documents];overall=aggregate(rows)
        overall.update(bootstrap_quality(rows,2000,c['seed']))
        overall['method']='document bootstrap stratified by domain; one training seed'
        for key in ['mlp_retained','routed_mlp_retained','label_recall','ideal_binary_input_row_bytes_per_token']:
            overall[key]=sum(r[key]*r['tokens'] for r in rows)/sum(r['tokens'] for r in rows)
        results[name]={'overall':overall,'domains':{d:aggregate([r for r in rows if r['domain']==d]) for d in ['web','code','conversation']}}
    recovery={}
    for keep in c['budgets']:
        pct=round(keep*100);f=results[f'fixed-{pct}']['overall'];o=results[f'oracle-{pct}']['overall']
        for method in methods:
            v=results[f'{method}-{pct}']['overall']
            recovery[f'{method}-{pct}']={metric:recovered_advantage(f[metric],v[metric],o[metric]) for metric in ['student_perplexity','student_nll','teacher_kl_nats']}
    # Paired resampling preserves covariance between fixed, predicted and oracle.
    rng=np.random.default_rng(c['seed']);groups=[[i for i,d in enumerate(documents) if d['metadata']['domain']==domain] for domain in ['web','code','conversation']]
    draws=np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(2000)])
    for keep in c['budgets']:
        pct=round(100*keep);boot={}
        for method in methods:
            values=np.array([[d['conditions'][f'{method}-{pct}'][k] for k in ['student_nll','kl','tokens']] for d in documents])
            sums=values[draws].sum(1)
            boot[method]={'student_nll':sums[:,0]/sums[:,2],'teacher_kl_nats':sums[:,1]/sums[:,2]}
            boot[method]['student_perplexity']=np.exp(boot[method]['student_nll'])
        for method in methods:
            row=recovery[f'{method}-{pct}'];intervals={}
            for metric in ['student_perplexity','student_nll','teacher_kl_nats']:
                denominator=boot['fixed'][metric]-boot['oracle'][metric];valid=denominator>1e-9
                values=(boot['fixed'][metric][valid]-boot[method][metric][valid])/denominator[valid]
                intervals[metric]={'ci95':np.quantile(values,[.025,.975]).tolist() if len(values) else None,
                    'valid_draws':int(valid.sum()),'total_draws':2000}
            row['paired_document_intervals']=intervals
    atomic_json(ROOT/'evaluation.json',{'status':'complete','results':results,'recovered_advantage':recovery,
        'documents':len(documents),'seconds':time.monotonic()-start,'checkpoints':checkpoint_hashes,
        'checks':{'all_on_bitwise_equal':True,'dense_prefix':True,'skipped_observations_hidden':True,'exact_budgets_except_threshold_teal':True},
        'interpretation':{'oracle':'greedy largest current activation, not true output-quality ceiling',
          'dense_oracle':'privileged dense-teacher target masks, reused on perturbed sparse trajectory',
          'previous_oracle':'privileged previous token dense target mask; first token uses fixed',
          'previous_selected':'exactly fixed by induction when initialized from fixed; no extra forward needed',
          'teal_mlp':'training-calibrated scalar magnitude threshold per MLP down input; variable realized retention; restricted reference, not full TEAL reproduction',
          'wina_mlp':'top-k activation times down-weight column norm; restricted reference, not full WINA reproduction',
          'headline':'(fixed error - predictor error)/(fixed error - greedy magnitude error); unclipped; undefined if denominator nonpositive'}})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','pilot','train','evaluate'])
    a=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.stage=='prepare':prepare()
    elif a.stage=='pilot':train(pilot=True)
    elif a.stage=='train':train()
    else:evaluate()


if __name__=='__main__':main()
