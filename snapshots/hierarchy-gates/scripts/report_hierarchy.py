"""Audit and report available hierarchy measurements; never fill missing runs."""
import argparse
import json
import math
from pathlib import Path
from scripts.report_matched_e2e import audit,paired_difference


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('artifacts/hierarchy-v1'))
    p.add_argument('--output',type=Path,default=Path('docs/hierarchy-results.md'));p.add_argument('--plot',action='store_true')
    a=p.parse_args();found={};lines=['# Hierarchical hidden-state gating pilot','',
      'Qwen3-1.7B, one training seed, fixed exploratory 63 documents. Attention and the first two MLPs stay dense. Compute is modeled batch-one decode FLOPs at 1K context, including the LM head and predictor; it is not measured latency.','',
      '| Variant | Budget label | Extra tokens | Calibration | Test total compute | Routed neurons kept | PPL change | KL | Top-1 |',
      '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for variant in 'ABCD':
        for budget in (50,40):
            for path in sorted((a.root/variant/f'budget-{budget}').glob('evaluation-*.json')):
                if path.stem.endswith(('-documents','-calibration-status')):continue
                result,rows=audit(path);tokens=int(path.stem.removeprefix('evaluation-'));o=result['overall']
                threshold=json.loads(path.with_name(f'threshold-{tokens:08d}.json').read_text())
                assert result['global_shift']==threshold['shift']
                assert result['base_offsets']==threshold['base_offsets']
                gap=threshold['validation_retention']-threshold['target_effective_retention']
                matched=threshold.get('matched_budget',True)
                if matched:
                    assert abs(gap)<=threshold['tolerance']
                else:
                    assert threshold['evaluation_calibration']=='unmatched_below_target' and -.001<=gap<=0
                    assert result['evaluation_calibration']=='unmatched_below_target' and not result['matched_budget']
                assert not {r['document'] for r in rows}&{r['document_sha256'] for r in threshold['validation_documents']}
                assert math.isclose(sum(r['total_compute']*r['tokens'] for r in rows)/o['tokens'],o['total_compute'],rel_tol=1e-10)
                assert abs(sum(o['levels'][k] for k in ('used','layer_saved','group_saved','neuron_saved'))-1)<1e-5
                found[variant,budget,tokens]=(result,rows,threshold)
                status='Matched' if matched else f'UNMATCHED: validation retention {100*gap:+.4f} pp'
                lines.append(f"| {variant} | {budget}% | {tokens:,} | {status} | {100*o['total_compute']:.3f}% | {100*o['routed_kept']:.3f}% | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    if not found:lines+=['','No held-out hierarchy results are available yet. Architecture tests and GPU preflight are not quality measurements.']
    comparisons={};lines+=['','## Paired comparisons with neuron-only A','',
      'The same validation total-compute target and extra training count are matched. Realized test compute differences are shown. Intervals describe document sampling, not seed variation.','',
      '| Comparison | Budget | Extra tokens | PPL gap (percentage points) | 95% interval | Test compute gap (pp) |',
      '| --- | ---: | ---: | ---: | --- | ---: |']
    for (variant,budget,tokens),(result,rows,t) in sorted(found.items()):
        if variant=='A' or ('A',budget,tokens) not in found:continue
        reference,rr,rt=found['A',budget,tokens]
        if not t.get('matched_budget',True) or not rt.get('matched_budget',True):
            lines.append(f'| {variant} − A | {budget}% | {tokens:,} | Not a matched-budget comparison | Calibration outside strict tolerance | — |')
            continue
        assert t['target_total_compute']==rt['target_total_compute']
        d=paired_difference(rows,rr);d['total_compute_gap_pp']=100*(result['overall']['total_compute']-reference['overall']['total_compute'])
        comparisons[f'{variant}-A-{budget}-{tokens}']=d;lo,hi=d['ppl_difference_ci95_pp']
        lines.append(f"| {variant} − A | {budget}% | {tokens:,} | {d['ppl_increase_difference_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {d['total_compute_gap_pp']:+.3f} |")
    lines+=['','## Exclusive savings within routed MLPs','',
      'Layer first, then groups within open layers, then neurons within open layer/groups. All values are shares of routed MLP arithmetic; attention savings are zero.','',
      '| Variant | Budget | Extra tokens | Layer saves | Group saves | Neuron saves | Kept |',
      '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for (v,b,n),(r,_,_) in sorted(found.items()):
        d=r['overall']['levels'];lines.append(f"| {v} | {b}% | {n:,} | {100*d['layer_saved']:.2f}% | {100*d['group_saved']:.2f}% | {100*d['neuron_saved']:.2f}% | {100*d['used']:.2f}% |")
    lines+=['','## Per-domain quality','',
      '| Variant | Budget | Extra tokens | Domain | Total compute | PPL change | KL |',
      '| --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for (v,b,n),(r,_,_) in sorted(found.items()):
        for domain,o in r['domains'].items():lines.append(f"| {v} | {b}% | {n:,} | {domain} | {100*o['total_compute']:.3f}% | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} |")
    if a.plot and found:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4.5))
        for variant in 'ABCD':
            rows=[]
            for budget in (50,40):
                available=[k for k in found if k[0]==variant and k[1]==budget]
                if not available:continue
                key=max(available,key=lambda k:k[2]);o=found[key][0]['overall']
                rows.append((100*o['total_compute'],100*o['relative_ppl_increase'],key[2]))
            if rows:
                rows.sort();ax.plot([x[0] for x in rows],[x[1] for x in rows],marker='o',label=variant)
                for x,y,n in rows:ax.annotate(f'{n/1e6:g}M',(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Modeled total decode FLOPs (% of dense Qwen)',ylabel='Perplexity increase (%)',title='Hierarchy pilot: latest available checkpoints')
        positive=[r[0]['overall']['relative_ppl_increase'] for r in found.values()]
        if min(positive)>0 and max(positive)/min(positive)>20:
            ax.set_yscale('log');ax.set_ylabel('Perplexity increase (%, logarithmic scale)')
        ax.grid(alpha=.2);ax.legend(title='Variant');fig.tight_layout()
        a.output.parent.mkdir(parents=True,exist_ok=True)
        for extension in ('png','pdf'):fig.savefig(a.output.with_suffix('.'+extension),dpi=180)
        plt.close(fig);lines+=['',f'![Quality versus modeled compute]({a.output.with_suffix(".png").name})','',
            'The plot connects at most two operating points per variant. Labels give the actual additional training token count; unmatched partial checkpoints do not establish architecture superiority.']
    skipped=[]
    for path in a.root.glob('*/budget-*/evaluation-*-calibration-status.json'):
        record=json.loads(path.read_text())
        if record['status']=='skipped_calibration_unavailable':skipped.append(str(path))
    if skipped:lines+=['','## Unevaluable checkpoints','',*[f'- {path}: no acceptable measured under-budget calibration point; quality was not evaluated.' for path in skipped]]
    native=a.root/'D/budget-50/native-final.json'
    if native.exists():
        r,rows=audit(native);o=r['overall']
        assert not r['matched_budget'] and r['global_shift']==0
        lines+=['','## D50 native-policy diagnostic','',
            f"The unchanged final D50 checkpoint at zero global shift has PPL increase {100*o['relative_ppl_increase']:+.3f}%, modeled total compute {100*o['total_compute']:.3f}%, routed retention {100*o['routed_kept']:.3f}%, and layer closure {100*o['levels']['layer_off']:.3f}%.",
            'This diagnostic receives no training updates and is not budget-matched. Compare its actual computation and layer closures with the original calibrated D50 result to assess calibration sensitivity.']
    lines+=['','Strict validation retention tolerance is 0.0005. Recovery may report a separately flagged point up to 0.001 below target; it is never included in matched-budget comparisons. Training and the global-shift policy are unchanged.']
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines)+'\n')
    a.output.with_suffix('.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    print(json.dumps({'audited_evaluations':len(found),'paired_comparisons':len(comparisons),'output':str(a.output)}))


if __name__=='__main__':main()
