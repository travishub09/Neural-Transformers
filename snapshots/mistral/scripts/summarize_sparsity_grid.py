"""Generate complete, plainly labelled comparison tables from saved results."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('artifacts/sparsity-calibration-v2'))
    p.add_argument('--output',type=Path,default=Path('docs/sparsity-calibration/grid-tables.md'))
    a=p.parse_args();r=json.loads((a.root/'grid-results.json').read_text())['results']
    rows=[('mlp',16),('mlp',32),('mlp',128),('attention',128),('both',16),('both',32),('both',128)]
    rules=['percentile','contribution','wina']
    s='# Component comparisons\n\nAll values use the same 63 documents. No protections in the first tables.\n'
    for metric,label in [('teacher_kl_nats','KL, nats/token'),('relative_ppl_increase','Perplexity change, percent')]:
        s+=f'\n## 50% requested retention: {label}\n\n| Component / MLP group size | Own percentile | Exact contribution | WINA-style |\n| --- | ---: | ---: | ---: |\n'
        for c,g in rows:
            vals=[r[f'{c}-g{g}-{rule}-k50-none']['overall'][metric] for rule in rules]
            if metric=='relative_ppl_increase':vals=[100*v for v in vals]
            name='Attention heads' if c=='attention' else f'{c} / {g}'
            s+='| '+name+' | '+' | '.join(f'{v:.4g}' for v in vals)+' |\n'
    s+='\n## Budget curves: exact contribution, no protections\n\n| Component / MLP group size | KL 90% | KL 75% | KL 50% | KL 25% | KL 10% |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for c,g in rows:
        vals=[r[f'{c}-g{g}-contribution-k{k}-none']['overall']['teacher_kl_nats'] for k in [90,75,50,25,10]]
        s+=f'| {c} / {g} | '+' | '.join(f'{v:.4g}' for v in vals)+' |\n'
    s+='\n## Protections: 50% requested, WINA-style, MLP groups of 32\n\nDense edge exemptions add retained computation. Always-on units count inside\nthe selection allowance. Percentages in the last columns are realized retention.\n\n| Component | Protection | PPL | KL | MLP retained | Heads retained |\n| --- | --- | ---: | ---: | ---: | ---: |\n'
    for c in ['mlp','attention','both']:
        g=128 if c=='attention' else 32
        for prot in ['none','always','edges','both']:
            v=r[f'{c}-g{g}-wina-k50-{prot}'];o=v['overall'];f=v['retained']
            s+=f"| {c} | {prot} | {o['student_perplexity']:.5g} | {o['teacher_kl_nats']:.4g} | {f['mlp']:.2%} | {f['attention']:.2%} |\n"
    s+='\n## Every evaluated condition\n\nThese are exploratory comparisons; confidence intervals are available in\n[the JSON report](grid-results.json). No simultaneous multiple-comparison claim.\n\n| Condition | PPL | PPL change | KL | MLP retained | Heads retained |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for name,v in r.items():
        o=v['overall'];f=v['retained'];s+=f"| {name} | {o['student_perplexity']:.6g} | {o['relative_ppl_increase']:.3%} | {o['teacher_kl_nats']:.6g} | {f['mlp']:.3%} | {f['attention']:.3%} |\n"
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(s)
    print(f'Wrote {len(r)} conditions to {a.output}')


if __name__=='__main__':main()
