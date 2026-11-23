"""Audit saved per-document sums and report all completed matched checkpoints."""
import json
import math
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def audit(path):
    result=read(path);rows=read(path.with_name(path.stem+'-documents.json'))
    total=sum(r['tokens'] for r in rows);o=result['overall']
    assert total==28857 and len(rows)==63
    assert len({r['document'] for r in rows})==63
    t=sum(r['teacher_nll'] for r in rows)/total;s=sum(r['student_nll'] for r in rows)/total
    expected={'dense_perplexity':math.exp(t),'student_perplexity':math.exp(s),
              'relative_ppl_increase':math.expm1(s-t),
              'teacher_kl_nats':sum(r['kl'] for r in rows)/total,
              'top1_agreement':sum(r['top1_agreement'] for r in rows)/total,
              'routed_kept':sum(r['routed_kept']*r['tokens'] for r in rows)/total}
    for k,v in expected.items():assert math.isclose(v,o[k],rel_tol=1e-10,abs_tol=1e-12),(path,k)
    for domain,summary in result['domains'].items():
        subset=[r for r in rows if r['domain']==domain];n=sum(r['tokens'] for r in subset)
        assert n==summary['tokens']
        assert math.isclose(math.expm1(sum(r['student_nll']-r['teacher_nll'] for r in subset)/n),
                            summary['relative_ppl_increase'],rel_tol=1e-10,abs_tol=1e-12)
    return result,rows


def paired_difference(a,b,repetitions=2000):
    """A minus B, paired by document; uncertainty is evaluation sampling only."""
    aa={r['document']:r for r in a};bb={r['document']:r for r in b}
    assert aa.keys()==bb.keys()
    ids=sorted(aa)
    for i in ids:
        assert aa[i]['domain']==bb[i]['domain'] and aa[i]['tokens']==bb[i]['tokens']
        assert math.isclose(aa[i]['teacher_nll'],bb[i]['teacher_nll'],rel_tol=1e-10)
    values=np.array([[aa[i]['teacher_nll'],aa[i]['student_nll'],bb[i]['student_nll'],
                      aa[i]['kl']-bb[i]['kl'],aa[i]['tokens'],
                      (aa[i]['routed_kept']-bb[i]['routed_kept'])*aa[i]['tokens']] for i in ids])
    groups=[np.array([j for j,i in enumerate(ids) if aa[i]['domain']==d])
            for d in ['web','code','conversation']]
    def statistic(v):
        t,a,b,k,n,r=v.sum(0)
        return np.array([100*(math.exp((a-t)/n)-math.exp((b-t)/n)),k/n,100*r/n])
    point=statistic(values);rng=np.random.default_rng(20261112)
    sampled=np.array([statistic(values[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])])
                      for _ in range(repetitions)])
    ci=np.quantile(sampled,[.025,.975],axis=0)
    return {'ppl_increase_difference_pp':float(point[0]),'ppl_difference_ci95_pp':ci[:,0].tolist(),
            'teacher_kl_difference':float(point[1]),'teacher_kl_difference_ci95':ci[:,1].tolist(),
            'retention_difference_pp':float(point[2]),'retention_difference_ci95_pp':ci[:,2].tolist(),
            'scope':'A minus B; stratified paired-document bootstrap, not seed uncertainty; no adjustment for retention differences.'}


def main():
    replicate=read(Path('configs/e2e-full.json')).get('replicate',1)
    lines=[f'# Matched end-to-end experiments — seed {replicate}','',
        'All completed continuation checkpoints are included. Results use the fixed exploratory 63-document set, 28,857 scored tokens. Calibration uses separate validation documents; actual test retention is reported.','',
        '## Calibrated pilot starting points','',
        '| Predictor | Target | Test kept | PPL increase | KL | Top-1 agreement |',
        '|---|---:|---:|---:|---:|---:|']
    for signal,root in [('Path','native-calibration-v1'),('Hidden','hidden-native-calibration-v1')]:
        for budget in [50,40]:
            p=Path(f'artifacts/{root}/evaluation-{budget}.json')
            if not p.exists():continue
            r,_=audit(p);o=r['overall']
            lines.append(f"| {signal} | {budget}% | {100*o['routed_kept']:.2f}% | {100*o['relative_ppl_increase']:+.2f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    lines+=['','## Full continuations','',
        '| Predictor | Target | Additional tokens | Test kept | PPL | PPL increase | KL | Top-1 agreement |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    collected=[]
    for signal,root in [('Path','artifacts/e2e-full-v1'),('Hidden','artifacts/e2e-hidden-full-v1')]:
        for budget in [50,40]:
            for path in sorted((Path(root)/f'budget-{budget}').glob('evaluation-*.json')):
                if path.stem.endswith('-documents'):continue
                r,rows=audit(path);o=r['overall'];label=path.stem.removeprefix('evaluation-')
                threshold=read(path.with_name(f'threshold-{label}.json'))
                assert abs(threshold['validation_retention']-budget/100)<=threshold['tolerance']
                assert not {x['document_sha256'] for x in threshold['validation_documents']} & {x['document'] for x in rows}
                assert r['offsets']==threshold['offsets']
                collected.append((signal,budget,int(label),r))
                lines.append(f"| {signal} | {budget}% | {int(label):,} | {100*o['routed_kept']:.2f}% | {o['student_perplexity']:.4f} | {100*o['relative_ppl_increase']:+.2f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    lines+=['','## Per-domain quality at each completed checkpoint','',
        '| Predictor | Target | Additional tokens | Domain | Test kept | PPL increase | KL |',
        '|---|---:|---:|---|---:|---:|---:|']
    for signal,budget,tokens,result in collected:
        for d,o in result['domains'].items():
            lines.append(f"| {signal} | {budget}% | {tokens:,} | {d} | {100*o['routed_kept']:.2f}% | {100*o['relative_ppl_increase']:+.2f}% | {o['teacher_kl_nats']:.5f} |")
    lines+=['','## Native training thresholds before calibration','',
        '| Predictor | Target | Additional tokens | Test kept | PPL increase | KL |',
        '|---|---:|---:|---:|---:|---:|']
    for signal,root in [('Path','artifacts/e2e-full-v1'),('Hidden','artifacts/e2e-hidden-full-v1')]:
        for budget in [50,40]:
            for path in sorted((Path(root)/f'budget-{budget}').glob('native-*.json')):
                if path.stem.endswith('-documents'):continue
                r,_=audit(path);o=r['overall'];tokens=int(path.stem.removeprefix('native-'))
                lines.append(f"| {signal} | {budget}% | {tokens:,} | {100*o['routed_kept']:.2f}% | {100*o['relative_ppl_increase']:+.2f}% | {o['teacher_kl_nats']:.5f} |")
    comparisons={}
    lines+=['','## Final path-versus-hidden comparison','']
    for budget in [50,40]:
        paths=[Path(f'artifacts/{root}/budget-{budget}/evaluation-10000000.json')
               for root in ['e2e-full-v1','e2e-hidden-full-v1']]
        if not all(p.exists() for p in paths):
            lines.append(f'{budget}%: both final checkpoints are not yet available.');continue
        _,a=audit(paths[0]);_,b=audit(paths[1]);d=paired_difference(a,b);comparisons[str(budget)]=d
        lines.append(f"At {budget}%, path minus hidden PPL increase is {d['ppl_increase_difference_pp']:+.3f} percentage points (95% document-bootstrap interval {d['ppl_difference_ci95_pp'][0]:+.3f} to {d['ppl_difference_ci95_pp'][1]:+.3f}). Test retention differs by {d['retention_difference_pp']:+.3f} percentage points; no retention adjustment was applied.")
    Path('docs/e2e-matched-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    lines+=['','## Token-level allocation at final checkpoints','',
        'Retention below describes neuron selection, not measured compute or speed. All attention runs; the first two MLP layers also run densely. The 5th and 95th percentiles describe allocation across tokens.','',
        '| Predictor | Target | Routed MLP kept | All MLP kept | Token 5th–95th percentile | Within-domain correlation with dense surprisal |',
        '|---|---:|---:|---:|---:|---:|']
    for signal,budget,tokens,result in collected:
        if tokens!=10000000:continue
        o=result['overall'];e=o['effort'];q=e['quantiles']
        lines.append(f"| {signal} | {budget}% | {100*o['routed_kept']:.2f}% | {100*o['all_mlp_kept']:.2f}% | {100*q['p05']:.2f}–{100*q['p95']:.2f}% | {e['pearson_within_domain_centered']:.4f} |")
    lines+=['','A broad allocation range alone does not establish useful adaptation to difficulty. Surprisal correlations are descriptive associations.','',
        '## Completed branch runtimes','',
        '| Predictor | Target | Additional tokens | Recorded hours | Effective tokens/second |',
        '|---|---:|---:|---:|---:|']
    for signal,root in [('Path','artifacts/e2e-full-v1'),('Hidden','artifacts/e2e-hidden-full-v1')]:
        for budget in [50,40]:
            p=Path(root)/f'budget-{budget}/training.json'
            if not p.exists():continue
            r=read(p)
            if r['status']!='complete':continue
            lines.append(f"| {signal} | {budget}% | {r['tokens']:,} | {r['seconds']/3600:.3f} | {r['tokens']/r['seconds']:.1f} |")
    lines+=['','These are recorded branch durations, including their in-run evaluations and checkpointing; they are not inference throughput. Account billing also includes setup, the hidden pilot, calibration, and time outside these branch timers.']
    cost_path=Path('artifacts/e2e-matched-v1/cost-summary.json')
    if cost_path.exists():
        c=read(cost_path)
        lines+=['','## GPU release and cost','',
            f"The pipeline completed in {c['pipeline_hours']:.2f} hours. Pod deletion was verified {c['seconds_completion_to_verified_release']:.1f} seconds after completion. All {c['verified_weight_backups']} recorded checkpoint backups were checksum-verified; the persistent volume was retained.",
            f"Creation-to-confirmed-release wall time was {c['wall_hours_creation_to_verified_release']:.2f} hours. At ${c['gpu_rate_usd_per_hour']:.2f}/hour, the GPU estimate is ${c['gpu_cost_wall_time_estimate_usd']:.2f}. Provider-reported pod charges retrieved at {c['billing_retrieved_utc']} totaled ${c['reported_total_pod_charges_usd']:.2f}; billing records lag, so this is not claimed as the final billed amount. Network-volume storage is separate."]
    accounting_path=Path('artifacts/neuron-guessing-v1/accounting.json')
    if accounting_path.exists():
        accounting=read(accounting_path)
        lines+=['','## Predictor overhead','',
            '| Predictor | Parameters | BF16 weight payload | Payload relative to Qwen | Projection FLOPs relative to dense 1K reference |',
            '|---|---:|---:|---:|---:|']
        for signal,label in [('binary','Path'),('hidden','Hidden')]:
            a=accounting['predictors'][signal]
            lines.append(f"| {label} | {a['parameters']:,} | {a['dense_bf16_predictor_weight_bytes']/1e6:.1f} MB | {100*a['relative_to_qwen_bf16_weight_payload']:.2f}% | {100*a['dense_projection_flop_fraction_1k']:.2f}% |")
        lines+=['','These are parameter/payload and arithmetic counts for the implemented dense projections, not measured memory traffic or decoding overhead. This path predictor is larger than the hidden-state predictor; this implementation does not establish a cheaper path-based router.']
    lines+=['','## References and limits','',
        'Dense PPL: 5.89295. Greedy current-magnitude references: +0.9568% at 50% and +2.390% at 40%. These are reference selectors, not proven quality ceilings. Native policies allocate different amounts across layers and tokens, while the reference uses uniform per-layer budgets.','',
        'Calibrated pilot starting points are reported above. The first-seed imitation-only binary/hidden results were +23.16%/+18.85% at 50% and +38.32%/+30.48% at 40%, using forced top-k. Comparing them directly mixes training and selection-policy changes.','',
        'Attention and the first two MLP layers remain dense. Retention excludes those layers and predictor overhead. Operations remain dense and masked; no real sparse inference speedup is demonstrated. Results from one training seed do not establish training-seed reproducibility.']
    out=Path('docs/e2e-matched-results.md');out.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'audited_calibrated_checkpoints':len(collected),'report':str(out)}))


if __name__=='__main__':main()
