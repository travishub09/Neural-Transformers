"""Dense-teacher activity oracles: true full-vocabulary KL and next-token PPL."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from safetensors.torch import load_file
from pathpred import gate_qwen3
from pathpred.collection import atomic_json, file_sha256
from pathpred.reordering import load_clustered_qwen
from pathpred.phase0 import budget_plan, select_units
from pathpred.oracle_checks import oracle_mask, mask_to_gates, quality_rows


def select_documents(records, split, per_domain, cfg):
    selected = []
    for domain in ['web', 'code', 'conversation']:
        docs = {}
        for r in records:
            if r['split'] == split and r['domain'] == domain and r['length'] >= cfg['minimum_sequence_length']:
                docs.setdefault(r['document_sha256'], r)
        ranked = sorted(docs, key=lambda d: hashlib.sha256((str(cfg['seed'])+d).encode()).hexdigest())
        if len(ranked) < min(8, per_domain):
            raise ValueError(f'Insufficient eligible {domain} documents: {len(ranked)}')
        selected += [docs[d] for d in ranked[:per_domain]]
    return selected


def aggregate(rows):
    tokens = sum(x['tokens'] for x in rows)
    totals = {key: sum(r[key] for r in rows) for key in ['teacher_nll', 'student_nll', 'kl', 'top1_agreement']}
    teacher, student = totals['teacher_nll']/tokens, totals['student_nll']/tokens
    return {'tokens': tokens, 'windows': len(rows), 'dense_nll': teacher, 'student_nll': student,
            'dense_perplexity': math.exp(teacher), 'student_perplexity': math.exp(student),
            'relative_ppl_increase': math.expm1(student-teacher), 'teacher_kl_nats': totals['kl']/tokens,
            'top1_agreement': totals['top1_agreement']/tokens}


def bootstrap_quality(rows, repetitions, seed):
    # Each row is one document after averaging random seeds. Stratify by domain.
    rng = np.random.default_rng(seed)
    domains = sorted({r['domain'] for r in rows})
    groups = [np.array([i for i,r in enumerate(rows) if r['domain'] == d]) for d in domains]
    values = np.array([[r['teacher_nll'], r['student_nll'], r['kl'], r['tokens']] for r in rows])
    samples = []
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        t,s,k,n = values[indices].sum(0)
        samples.append([math.expm1((s-t)/n), k/n])
    quantiles = np.quantile(samples, [.025, .975], axis=0)
    return {'relative_ppl_increase_ci95': quantiles[:,0].tolist(), 'teacher_kl_nats_ci95': quantiles[:,1].tolist(),
            'method': 'paired document bootstrap within domains; random-mask seeds averaged first'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--output', type=Path, default=Path('artifacts/oracle-quality-v1'))
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = json.loads(Path('configs/phase0-followup.json').read_text())
    root = Path('data/stage3-corpus-v1')
    cm = json.loads((root/'manifest.json').read_text())
    for entry in cm['files']:
        assert file_sha256(root/entry['file']) == entry['sha256']
    records = [json.loads(x) for x in (root/'sequences.jsonl').read_text().splitlines()]
    selected = select_documents(records, 'validation' if a.pilot else 'test',
        cfg['pilot_documents_per_domain'] if a.pilot else cfg['test_documents_per_domain'], cfg)
    fp = {str(f):file_sha256(Path(f)) for f in ['configs/phase0-followup.json',
        'docs/phase0-followup-protocol.md', 'scripts/oracle_quality.py', 'pathpred/oracle_checks.py', 'pathpred/phase0.py',
        'pathpred/gated_qwen.py', 'pathpred/reordering.py', 'data/stage5-signals-v1/thresholds.safetensors',
        'data/stage3-corpus-v1/manifest.json', 'data/stage6-phase0-v1/manifest.json']}
    fp['pilot'] = a.pilot
    manifest_path = a.output/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert previous['fingerprint'] == fp
        if previous['status'] == 'complete':
            print('Already complete', flush=True); return
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {'status':'in_progress','fingerprint':fp,'config':cfg,
                'selected_documents':selected,'unit_group_size':128,'dense_prefix_layers':2}
    atomic_json(manifest_path, manifest)
    tokens = load_file(str(root/'tokens.safetensors'))['input_ids']
    obs = load_file('data/stage6-phase0-v1/examples.safetensors')
    train = (obs['split_codes'] == 0) & (obs['previous_sample_indices'] >= 0)
    static = {'fixed_frequency':obs['binary'][train].float().mean(0).cuda(),
              'fixed_mean_norm':obs['shares'][train].mean(0).cuda()}
    del obs
    q = load_file('data/stage5-signals-v1/thresholds.safetensors', device='cuda')['q75']
    costs = json.loads(Path('data/stage4-clusters-v1/unit-costs.json').read_text())
    plans = {str(b):budget_plan(costs,b,0,0) for b in cfg['budgets']}
    manifest['plans'] = plans
    model = load_clustered_qwen('checkpoints/qwen3-1.7b-base-clustered-v1', dtype=torch.bfloat16,
        device_map={'':'cuda'}, attn_implementation='eager', local_files_only=True)
    controller = gate_qwen3(model,128)
    start = time.monotonic()
    all_documents = []
    with torch.inference_mode():
        for number, meta in enumerate(selected):
            output_path = a.output / f"window-{meta['sequence_id']}.json"
            if output_path.exists():
                saved=json.loads(output_path.read_text()); assert saved['fingerprint']==fp
                all_documents.append(saved); continue
            if time.monotonic()-start > cfg['maximum_runtime_hours']*3600:
                raise TimeoutError('Oracle evaluation runtime guard')
            length=min(meta['length'],cfg['maximum_sequence_length'])
            ids=tokens[meta['sequence_id']:meta['sequence_id']+1,:length].long().cuda()
            with controller.use_gates(record=True) as trace:
                teacher=model(ids,use_cache=False).logits
            norms=torch.stack([torch.cat([trace.records[(l,k)].observed_norms[0]
                for k in ['attention','mlp']],-1) for l in range(28)],1)
            all_on=torch.ones_like(norms,dtype=torch.bool)
            with controller.use_gates(mask_to_gates(all_on)):
                check=model(ids,use_cache=False).logits
            assert torch.equal(teacher,check), 'All-on forward not identical'
            del check,trace
            conditions=[('dense',None,0,all_on),
                ('literal_q75',None,0,oracle_mask(norms,q,mode='threshold'))]
            for b in cfg['budgets']:
                if a.pilot and b != .25: continue
                counts=plans[str(b)]['counts']
                conditions.append(('oracle_ranked',b,0,oracle_mask(norms,q,counts,mode='ranked')))
                for name,scores in static.items():
                    mask=all_on.clone()
                    mask[:,2:]=select_units(scores[None,2:].expand(length,-1,-1),counts)
                    conditions.append((name,b,0,mask))
                for seed in cfg['mask_seeds']:
                    if a.pilot and seed != cfg['mask_seeds'][0]:continue
                    actual_seed=seed*100000+meta['sequence_id']
                    conditions.append(('oracle_binary',b,seed,oracle_mask(norms,q,counts,mode='binary',seed=actual_seed)))
                    conditions.append(('random',b,seed,oracle_mask(norms,q,counts,mode='random',seed=actual_seed)))
            evaluated=[]
            for name,budget,seed,mask in conditions:
                if name == 'dense': student=teacher
                else:
                    with controller.use_gates(mask_to_gates(mask)):
                        student=model(ids,use_cache=False).logits
                metrics=quality_rows(teacher,student,ids)
                row={k:float(v.double().sum()) for k,v in metrics.items()}
                row.update(name=name,budget=budget,seed=seed,tokens=length-1,domain=meta['domain'],
                           document=meta['document_sha256'],max_token_kl=float(metrics['kl'].max()))
                selected_heads=mask[...,:16].float().sum((1,2))
                selected_groups=mask[...,16:].float().sum((1,2))
                reference=costs['scenarios']['1024']
                mandatory=reference['mandatory_kv_projection_flops']+reference['mandatory_lm_head_flops']
                head_cost=4*2048*128+4*torch.arange(1,length+1,device='cuda')*128
                group_cost=6*2048*128
                real_cost=mandatory+selected_heads*head_cost+selected_groups*group_cost
                dense_cost=mandatory+28*16*head_cost+28*48*group_cost
                row['selected_units_fraction']=float(mask.float().mean())
                row['selected_head_fraction']=float(mask[...,:16].float().mean())
                row['selected_mlp_fraction']=float(mask[...,16:].float().mean())
                row['hypothetical_causal_matmul_fraction']=float(real_cost.double().sum()/dense_cost.double().sum())
                row['reference_1k_matmul_fraction']=float((mandatory+(selected_heads+selected_groups).mean()*group_cost)/reference['dense_accounted_flops'])
                evaluated.append(row)
                if name != 'dense':del student
            saved={'fingerprint':fp,'metadata':meta,'length':length,'all_on_bitwise_equal':True,'conditions':evaluated}
            atomic_json(output_path,saved);all_documents.append(saved)
            print(json.dumps({'completed_windows':number+1,'total_windows':len(selected),'seconds':time.monotonic()-start}),flush=True)
            del teacher,norms,conditions
    grouped={}
    for document in all_documents:
        for row in document['conditions']:
            key=row['name']+('' if row['budget'] is None else '-b'+str(round(row['budget']*100)))
            grouped.setdefault(key,{}).setdefault(row['document'],[]).append(row)
    results={}
    for key,documents in grouped.items():
        averaged=[]
        seed_rows={}
        for rows in documents.values():
            base=dict(rows[0])
            for metric in ['teacher_nll','student_nll','kl','top1_agreement','selected_units_fraction',
                           'selected_head_fraction','selected_mlp_fraction','hypothetical_causal_matmul_fraction','reference_1k_matmul_fraction']:
                base[metric]=sum(r[metric] for r in rows)/len(rows)
            averaged.append(base)
            for r in rows:seed_rows.setdefault(str(r['seed']),[]).append(r)
        overall=aggregate(averaged)
        overall.update(bootstrap_quality(averaged,cfg['bootstrap_repetitions'],cfg['seed']))
        domains={domain:aggregate([r for r in averaged if r['domain']==domain]) for domain in ['web','code','conversation']}
        quality_pass=all(v['relative_ppl_increase']<=cfg['quality_max_relative_ppl_increase'] and
                         v['teacher_kl_nats']<=cfg['quality_max_teacher_kl_nats'] for v in [overall,*domains.values()])
        results[key]={'overall':overall,'domains':domains,'seeds':{s:aggregate(rs) for s,rs in seed_rows.items()},
            'quality_pass_provisional':quality_pass,
            'budget_observed':{m:sum(r[m]*r['tokens'] for r in averaged)/sum(r['tokens'] for r in averaged)
                for m in ['selected_units_fraction','selected_head_fraction','selected_mlp_fraction','hypothetical_causal_matmul_fraction','reference_1k_matmul_fraction']}}
    report={'status':'complete','pilot':a.pilot,'results':results,'fingerprint':fp,'config':cfg,
        'documents':len(all_documents),'all_on_bitwise_equal':all(x['all_on_bitwise_equal'] for x in all_documents),
        'seconds':time.monotonic()-start,'notes':['Teacher activity oracle is not a quality-optimal mask',
            'Full-vocabulary forward KL and real-token teacher-forced PPL; no sparse kernels or generation-speed claim',
            'Binary exact-budget oracle uses random tie-breaking; ranked oracle sees additional magnitudes',
            'Initial two layers and KV are always dense; literal q75 budget varies',
            'Same existing test documents: exploratory follow-up, not independent replication']}
    atomic_json(a.output/'results.json',report)
    manifest.update(status='complete',seconds=report['seconds']);atomic_json(manifest_path,manifest)
    print(json.dumps({'status':'complete','documents':len(all_documents),'seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()
