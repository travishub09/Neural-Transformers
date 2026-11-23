"""Audit available exact-history planner results; never fill in pending rows."""
import argparse
import json
from pathlib import Path
from scripts.report_matched_e2e import audit, paired_difference
from pathpred.collection import atomic_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('artifacts/planner-v1'))
    parser.add_argument('--output',type=Path,default=Path('docs/planner-results.md'));args=parser.parse_args()
    labels={'autoregressive':'Autoregressive + previous path','one_shot_history':'One-shot + previous path',
            'one_shot_hidden':'One-shot hidden only','references/binary':'Closed-loop path (existing weights)',
            'references/hidden':'Closed-loop hidden (existing weights)'}
    lines=['# Whole-path planner pilot','',
        'Exact sequential language-token histories; independent documents batched. All gates are hard at evaluation. '
        'Validation-only thresholds target routed MLP neuron retention; attention and the first two layers remain dense. '
        'Dense masking is not an inference speedup.','',
        '| Method | Target | Test kept | PPL | PPL change | KL | Top-1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    results={};comparisons={}
    for budget in (50,40):
        for name,label in labels.items():
            root=args.root/name/f'budget-{budget}';path=root/'evaluation.json'
            if not path.exists():continue
            result,rows=audit(path);threshold=json.loads((root/'threshold.json').read_text())
            assert abs(threshold['validation_retention']-budget/100)<=threshold['tolerance']
            assert threshold['offsets']==result['offsets']
            assert not {r['document_sha256'] for r in threshold['identity']['validation_documents']}&{r['document'] for r in rows}
            results[name,budget]=(result,rows);o=result['overall']
            lines.append(f"| {label} | {budget}% | {100*o['routed_kept']:.3f}% | {o['student_perplexity']:.4f} | "
                f"{100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    if not results:lines+=['','No completed quality results yet. Architecture and smoke tests are not quality results.']
    lines+=['','## Paired comparisons','',
        'A minus B; negative PPL gaps favor A. Document-bootstrap intervals do not measure training-seed uncertainty. '
        'The primary matched-training comparison is autoregressive versus same-input one-shot. '
        'Existing closed-loop weights have different training histories.','',
        '| A vs B | Target | PPL gap (pp) | 95% interval | Kept gap (pp) |',
        '|---|---:|---:|---|---:|']
    for budget in (50,40):
        for a,b in [('autoregressive','one_shot_history'),('autoregressive','one_shot_hidden'),
                    ('autoregressive','references/hidden'),('autoregressive','references/binary')]:
            if (a,budget) not in results or (b,budget) not in results:continue
            result=paired_difference(results[a,budget][1],results[b,budget][1]);comparisons[f'{a}:{b}:{budget}']=result
            lo,hi=result['ppl_difference_ci95_pp']
            lines.append(f"| {labels[a]} vs {labels[b]} | {budget}% | {result['ppl_increase_difference_pp']:+.3f} | "
                         f"[{lo:+.3f}, {hi:+.3f}] | {result['retention_difference_pp']:+.3f} |")
    lines+=['','## Previous-path ablation','',
        '| Target | Policy | Test kept | PPL change | KL |',
        '|---|---|---:|---:|---:|']
    for budget in (50,40):
        for policy in ('fixed','matched'):
            path=args.root/'autoregressive'/f'budget-{budget}'/f'no-previous-{policy}.json'
            if not path.exists():continue
            result,_=audit(path);o=result['overall']
            lines.append(f"| {budget}% | {policy} | {100*o['routed_kept']:.3f}% | "
                         f"{100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} |")
    lines+=['','Training histories are approximate two-pass histories; evaluation uses actual sequential histories. '
        'Per-domain and per-layer magnitude-reference accuracy are preserved in each evaluation JSON. '
        'Predictor sizes and GPU smoke timings are reported separately in gpu-smoke.json.']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text('\n'.join(lines)+'\n')
    atomic_json(args.output.with_suffix('.json'),comparisons)


if __name__=='__main__':main()
