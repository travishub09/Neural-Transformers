"""Report saved quality measurements; never executes or trains a model."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('artifacts/neuro-quality-v1'))
    args=parser.parse_args();root=args.root
    d=json.loads((root/'results.json').read_text())
    lines=['# NeuroPrefetcher perplexity follow-up','',
        'The user explicitly requested language-model quality evaluation despite low centroid-ID accuracy. This run reused the saved 13M-token predictor and performed no training updates. It did not rerun the planner.','',
        '## What the paper measures','',
        'Section 4.7 describes approximately 99% correct centroid selections per layer decision, then separately evaluates WikiText-2 perplexity and HellaSwag accuracy. The released predictor code computes exact argmax-ID accuracy. Its released training script uses the same dataset object for training and evaluation loaders; the paper does not fully specify the split behind the 99% figure. This does not establish that the published results used that exact evaluation setup.','',
        'The released A100 quick-check PPL script feeds true tokens sequentially, scores next-token log probabilities, and exponentiates mean negative log probability. It uses the first 2,048 WikiText-2 test tokens and divides 2,047 scored losses by 2,048. Our matched comparison uses the actual scored-position count and the planner’s 63 documents. These absolute PPL values are not directly comparable to the paper’s model/dataset numbers.','',
        'Sources: [paper §4.7](https://arxiv.org/html/2608.22643v1#S4.SS7), [centroid accuracy code](https://github.com/nobeldhar/NeuroPrefetcher/blob/4c47f08b37e5b498ac0056c874b7aea45fc616fa/training/predictor/train_gate_up_predictor_updated.py#L175), [released quick-check PPL code](https://github.com/nobeldhar/NeuroPrefetcher/blob/4c47f08b37e5b498ac0056c874b7aea45fc616fa/quality_study/engine/eval_ppl_a100.py#L76).','',
        '## Matched-document quality','',
        '| Model / policy | Test neurons kept | Dense PPL | Sparse PPL | PPL increase | KL, nats | Top-token agreement |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,r in d['saved_planner'].items():
        o=r['overall'];lines.append(f"| Saved planner / {label}% validation target | {100*o['routed_kept']:.3f}% | {o['dense_perplexity']:.4f} | {o['student_perplexity']:.4f} | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    for label,r in d['results'].items():
        o=r['overall'];lines.append(f"| NeuroPrefetcher / {label} | {100*o['routed_kept']:.3f}% | {o['dense_perplexity']:.4f} | {o['student_perplexity']:.4f} | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    lines+=['',
        'Labels 0.5000/0.4000/0.5162 are validation intersection-retention targets. All use the same nominal 50%-vocabulary checkpoint. The 40% row is a calibrated operating point, not a separately trained 40% predictor. Native uses the original centroid masks unchanged; intersecting two 50%-dense masks does not imply 50% effective retention. The classifier and dense-prefix centroid-ID vocabulary are fixed throughout.','',
        '| Neuro policy | Domain | Dense PPL | Sparse PPL | PPL increase | KL | Agreement |','|---|---|---:|---:|---:|---:|---:|']
    for label,r in d['results'].items():
        for domain,o in r['domains'].items():
            lines.append(f"| {label} | {domain} | {o['dense_perplexity']:.4f} | {o['student_perplexity']:.4f} | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    lines+=['','## Paired document-bootstrap differences','',
        'NeuroPrefetcher minus saved planner; 2,000 resamples stratified by domain. Intervals describe document uncertainty, not training-seed variation. The 0.5162 comparison is a sensitivity comparison against the planner’s 50%-validation target, not an identical calibration target.','',
        '| Neuro target | Same validation target? | PPL-increase difference, percentage points | 95% interval | KL-difference95% interval | Agreement-difference95% interval, pp |',
        '|---|---|---:|---|---|---|']
    for label,p in d['paired'].items():
        base=str(round(100*p['planner_validation_target']));gap=100*(d['results'][label]['overall']['relative_ppl_increase']-d['saved_planner'][base]['overall']['relative_ppl_increase'])
        ci=[100*x for x in p['ppl_relative_increase_difference_ci95']];ki=p['kl_nats_difference_ci95'];ai=[100*x for x in p['top1_difference_ci95']]
        lines.append(f"| {label} | {p['same_validation_target']} | {gap:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | [{ki[0]:+.5f}, {ki[1]:+.5f}] | [{ai[0]:+.2f}, {ai[1]:+.2f}] |")
    lines+=['','## Centroid accuracy and temporal reuse','',
        '| Policy | Gate-ID accuracy | Up-ID accuracy | Previous-token reuse, count-weighted | Previous-token reuse, token mean |',
        '|---|---:|---:|---:|---:|']
    for label,r in d['results'].items():
        o=r['overall'];lines.append(f"| {label} | {100*o['gate_centroid_top1']:.2f}% | {100*o['up_centroid_top1']:.2f}% | {100*o['previous_token_reuse_count_weighted']:.2f}% | {100*o['previous_token_reuse_token_mean']:.2f}% |")
    lines+=['','## Calibration details','',
        'The calibrated settings change each centroid mask’s density while keeping the learned classifier IDs fixed. This is an explicit adaptation for matching effective retention, not a newly trained codebook at each sparsity. Native is the unmodified-policy control.','',
        '| Target | Neurons per gate/up centroid | Validation intersection kept | Matched within tolerance? |',
        '|---|---:|---:|---|']
    for label,r in d['results'].items():
        if label=='native':continue
        c=r['calibration'];lines.append(f"| {label} | {c['count_per_centroid']} / 6144 | {100*c['validation_retention']:.4f}% | {c['matched_budget']} |")
    lines+=['','## Integrity and scope','',
        'Same frozen Qwen3-1.7B checkpoint, first 2 layers dense, all attention dense, same 63 document IDs/scoring windows and independent cached teacher/student executions. Dense-prefix features are identical for teacher and student. Calibration uses validation only; saved planner scores are read-only. Each inactive intermediate neuron is masked before the down projection, modeling gate/up rows and down columns skipped. Dense masking is a quality experiment and establishes no kernel speedup.']
    for name in ('cost','cleanup','verification'):
        path=root/(name+'.json')
        if path.exists():d[name]=json.loads(path.read_text())
    if 'cost' in d:
        cost=d['cost'];lines+=['',f"Completed GPU allocation: {cost['allocation_minutes']:.2f} minutes at ${cost['hourly_rate']:.2f}/hour; runtime-based GPU cost ${cost['runtime_cost_usd']:.4f}. Provider posting may lag; shared storage is excluded."]
    if 'cleanup' in d:lines+=['','GPU shutdown verified; the persistent checkpoint volume was retained.']
    Path('docs/neuro-quality-results.md').write_text('\n'.join(lines)+'\n')
    Path('docs/neuro-quality-results.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({'evaluations':len(d['results']),'paired':len(d['paired'])}))


if __name__=='__main__':main()
