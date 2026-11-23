"""Render measured controls; missing results remain explicitly missing."""
import json
from pathlib import Path


def main():
    root=Path('artifacts/neuro-controls-v1');mistral=Path('artifacts/neuro-mistral-v1')
    lines=['# NeuroPrefetcher controls',''];bundle={}
    oracle=root/'oracle/evaluation.json'
    lines+=['## No-predictor Qwen control','']
    if oracle.exists():
        o=json.loads(oracle.read_text());old=json.loads(Path('artifacts/neuro-quality-v1/evaluation-0.5000.json').read_text());bundle['oracle']=o
        lines+=['Same 63 held-out documents, dense first two layers, attention fully on. Validation target: 50% of suffix neurons kept. Dense Qwen perplexity: 5.8934.','',
            '| Selection | Test kept | PPL | PPL increase | KL | Top-token agreement |','|---|---:|---:|---:|---:|---:|']
        for name,r in [('Trained centroid predictor',old['overall']),('True nearest patterns; no predictor',o['overall'])]:
            lines.append(f"| {name} | {r['routed_kept']:.3%} | {r['student_perplexity']:.4f} | {r['relative_ppl_increase']:+.2%} | {r['teacher_kl_nats']:.4f} | {r['top1_agreement']:.2%} |")
        lines+=['','Nearest patterns are a privileged label control, not a globally optimal output-quality ceiling. Labels and vocabulary construction remain possible sources of damage.','']
        lines+=['| Domain | Dense PPL | Nearest-pattern PPL | PPL increase | KL |','|---|---:|---:|---:|---:|']
        for name,r in o['domains'].items():
            lines.append(f"| {name} | {r['dense_perplexity']:.4f} | {r['student_perplexity']:.4f} | {r['relative_ppl_increase']:+.2%} | {r['teacher_kl_nats']:.4f} |")
        lines.append('')
        paired=root/'oracle/paired-vs-predictor.json'
        if paired.exists():
            bounds=json.loads(paired.read_text())['ppl_relative_increase_difference_ci95']
            lines += [f'Nearest-pattern minus trained-predictor PPL-increase difference: paired 95% interval {100*bounds[0]:.2f} to {100*bounds[1]:.2f} percentage points. This interval includes zero.','']
    else:lines+=['Not completed.','']
    pair=root/'relaxed-qwen/results.json';lines+=['## Qwen with 20 dense layers','']
    if pair.exists():
        d=json.loads(pair.read_text());bundle['relaxed_qwen']=d;lines+=['Saved checkpoints, no additional training. Both retain their original layer-2 planning context. Eight sparse suffix MLPs; separately calibrated to 50% suffix retention. This is approximately 85.7% of all MLP neurons retained, not 50% of total model compute.','',
            '| Method | Test kept in suffix | PPL | PPL increase | KL | Top-token agreement |','|---|---:|---:|---:|---:|---:|']
        for name,key in [('Centroid predictor','neuro'),('Upfront planner','planner')]:
            r=d[key]['overall'];lines.append(f"| {name} | {r['routed_kept']:.3%} | {r['student_perplexity']:.4f} | {r['relative_ppl_increase']:+.2%} | {r['teacher_kl_nats']:.4f} | {r['top1_agreement']:.2%} |")
        bounds=d['paired']['ppl_relative_increase_difference_ci95'];lines+=['',f'Paired document-bootstrap 95% interval, centroid minus planner PPL-increase difference: {100*bounds[0]:.3f} to {100*bounds[1]:.3f} percentage points.','']
        lines+=['| Domain | Centroid PPL increase | Planner PPL increase |','|---|---:|---:|']
        for domain in ('web','code','conversation'):
            lines.append(f"| {domain} | {d['neuro']['domains'][domain]['relative_ppl_increase']:+.2%} | {d['planner']['domains'][domain]['relative_ppl_increase']:+.2%} |")
        lines.append('')
    else:lines+=['Not completed.','']
    lines+=['## Released Mistral configuration','']
    p=mistral/'results.json'
    if p.exists():
        d=json.loads(p.read_text());bundle['released_mistral']=d;lines+=['Exact released 2,048-token WikiText slice; s30 assets, advertised as 50% active. The reference script divides NLL by 2,048 despite scoring 2,047 next-token predictions.','',
            '| Execution | Released PPL | Our PPL, same convention | Corrected PPL | Suffix kept | KL |','|---|---:|---:|---:|---:|---:|']
        for key,name,expected in [('32','Dense','3.9937'),('20','20 dense + 12 sparse, released engine','5.6903'),('20_full_down','Same masks; explicitly full down projection','—')]:
            if key not in d:continue
            r=d[key];lines.append(f"| {name} | {expected} | {r['paper_denominator_ppl']:.4f} | {r['correct_denominator_ppl']:.4f} | {r['suffix_kept']:.3%} | {r['kl']:.4f} |")
        lines+=['','This validates released assets/runtime only to the extent that the measured numbers match. It does not validate every adaptation in the separately trained Qwen predictor. No Mistral-trained planner is available, so this is not a paired Mistral comparison.','']
    else:lines+=['Not completed.','']
    lines+=['## Interpretation limits','',
        'Full down multiplication on a masked intermediate is algebraically equivalent to selected down columns. Its compute accounting differs, but it does not recover dropped information.',
        'The released Mistral model uses shared gate/up banks, while the prior Qwen adaptation uses per-layer banks. Dense-prefix fractions also differ between 28-layer Qwen and 32-layer Mistral.',
        'These are quality controls, not measured sparse speedups.','']
    if (root/'cost.json').exists() and (mistral/'cost.json').exists():
        costs=[json.loads((r/'cost.json').read_text()) for r in (root,mistral)]
        interrupted=json.loads((root/'interrupted-provision.json').read_text())['gpu_runtime_charge_upper_bound']
        total=sum(r['gpu_runtime_charge_estimate'] for r in costs)+interrupted
        bundle['cost']={'completed_allocations':costs,'interrupted_setup_upper_bound':interrupted,'gpu_total_estimate':total}
        lines+=['## Cost and shutdown','',f'Estimated GPU allocation charges: ${total:.3f}, including up to ${interrupted:.3f} for the interrupted first setup. Based on allocation durations and hourly rates; provider billing has not yet posted. Storage is excluded. Both evaluation GPUs were deleted and absence verified; the existing network volume was preserved.','']
    Path('docs/neuro-controls-results.md').write_text('\n'.join(lines))
    Path('docs/neuro-controls-results.json').write_text(json.dumps(bundle,indent=2)+'\n')

if __name__=='__main__':main()
