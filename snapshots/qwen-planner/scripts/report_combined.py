"""Report all available matched checkpoints; never substitute missing results."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.report_matched_e2e import audit, paired_difference


def comparison(a, b):
    result = paired_difference(a, b)
    aa = {r['document']: r for r in a}; bb = {r['document']: r for r in b}
    keys = sorted(aa)
    values = np.array([[aa[k]['top1_agreement']-bb[k]['top1_agreement'], aa[k]['tokens']] for k in keys])
    groups = [np.array([i for i,k in enumerate(keys) if aa[k]['domain'] == d])
              for d in ('web','code','conversation')]
    def statistic(v):
        difference, tokens = v.sum(0)
        return 100*difference/tokens
    rng = np.random.default_rng(20261112)
    sampled = [statistic(values[np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])])
               for _ in range(2000)]
    result['top1_difference_pp'] = float(statistic(values))
    result['top1_difference_ci95_pp'] = np.quantile(sampled, [.025, .975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('artifacts/combined-v1'))
    parser.add_argument('--output', type=Path, default=Path('docs/combined-results.md'))
    args = parser.parse_args()
    labels = {'combined': 'Combined', 'binary': 'Path only', 'hidden': 'Hidden only'}
    lines = ['# Combined predictor experiment', '',
        'Exploratory 63-document evaluation, 28,857 scored tokens. Thresholds selected on separate validation documents. All kept fractions concern the 26 routed MLP layers; attention and the first two layers stay dense. Dense masked simulation, with no sparse speedup claimed.', '',
        '| Model | Target | Extra tokens per branch | Test kept | PPL | PPL change | KL | Top-1 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    collected = {}
    for budget in (50, 40):
        for signal, label in labels.items():
            for path in sorted((args.root/signal/f'budget-{budget}').glob('evaluation-*.json')):
                if path.stem.endswith('-documents'): continue
                result, rows = audit(path); o = result['overall']
                tokens = int(path.stem.removeprefix('evaluation-'))
                threshold = json.loads(path.with_name(f'threshold-{tokens:08d}.json').read_text())
                assert abs(threshold['validation_retention']-budget/100) <= threshold['tolerance']
                assert threshold['offsets'] == result['offsets']
                assert not {r['document_sha256'] for r in threshold['validation_documents']} & {r['document'] for r in rows}
                collected[signal,budget,tokens] = result,rows
                lines.append(f"| {label} | {budget}% | {tokens:,} | {100*o['routed_kept']:.3f}% | {o['student_perplexity']:.4f} | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    comparisons = {}
    lines += ['', '## Matched checkpoint comparisons', '',
        'Differences are A minus B. Negative PPL differences favor A. Intervals are paired document-bootstrap intervals, not training-seed uncertainty.', '',
        '| A vs B | Target | Extra tokens | PPL gap (pp) | 95% interval | Test kept gap (pp) |',
        '|---|---:|---:|---:|---|---:|']
    for budget in (50,40):
        for tokens in (2_000_000,5_000_000,10_000_000):
            for a,b in (('combined','hidden'),('combined','binary'),('binary','hidden')):
                if (a,budget,tokens) not in collected or (b,budget,tokens) not in collected: continue
                d = comparison(collected[a,budget,tokens][1], collected[b,budget,tokens][1])
                comparisons[f'{a}-{b}-{budget}-{tokens}'] = d
                lo,hi = d['ppl_difference_ci95_pp']
                lines.append(f"| {labels[a]} vs {labels[b]} | {budget}% | {tokens:,} | {d['ppl_increase_difference_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {d['retention_difference_pp']:+.3f} |")
    lines += ['', '## Per-domain results', '',
        '| Model | Target | Extra tokens | Domain | Test kept | PPL change | KL | Top-1 |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for (signal,budget,tokens),(result,_) in sorted(collected.items()):
        for domain,o in result['domains'].items():
            lines.append(f"| {labels[signal]} | {budget}% | {tokens:,} | {domain} | {100*o['routed_kept']:.3f}% | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    lines += ['', '## Combined reliance checks', '',
        '`no_hidden` zeros the content input. `zero_bits` zeros current and previous firing bits but preserves observed flags. `no_path` removes current and previous path embeddings and their observation flags. Fixed-policy tests keep the intact model’s threshold; matched tests recalibrate on validation only. Actual kept fractions must be considered in either case.', '',
        '| Target | Ablation | Policy | Test kept | PPL change | KL | Top-1 |',
        '|---|---|---|---:|---:|---:|---:|']
    for budget in (50,40):
        for mode in ('no_hidden','zero_bits','no_path'):
            for policy in ('fixed','matched'):
                path = args.root/'combined'/f'budget-{budget}'/f'reliance-{mode}'/f'{policy}.json'
                if not path.exists(): continue
                result,_ = audit(path); o=result['overall']
                lines.append(f"| {budget}% | {mode} | {policy} | {100*o['routed_kept']:.3f}% | {100*o['relative_ppl_increase']:+.3f}% | {o['teacher_kl_nats']:.5f} | {100*o['top1_agreement']:.2f}% |")
    if not collected:
        lines += ['', '**No training/evaluation results exist yet.** Architecture tests are not quality results.']
    lines += ['', 'Equal data and optimizer steps do not mean equal predictor size or inference cost. Combined-versus-hidden results and the reliance interventions must be interpreted together; this is not a parameter-matched comparison.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n')
    args.output.with_suffix('.json').write_text(json.dumps(comparisons, indent=2)+'\n')


if __name__ == '__main__': main()
