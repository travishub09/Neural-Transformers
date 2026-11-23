"""Independently reconstruct quality results from document-level outputs."""
import json
import math
from pathlib import Path
import shutil
import hashlib
import numpy as np


def main():
    source=Path('artifacts/neuron-guessing-v1');out=Path('docs/neuron-guessing');out.mkdir(parents=True,exist_ok=True)
    evaluation=json.loads((source/'evaluation.json').read_text())
    documents=[json.loads(p.read_text()) for p in source.glob('eval-window-*.json')]
    assert len(documents)==evaluation['documents']
    for signal,expected in evaluation['checkpoints'].items():
        with (source/f'{signal}.safetensors').open('rb') as checkpoint:
            digest=hashlib.file_digest(checkpoint,'sha256').hexdigest()
        assert digest==expected, signal
    for name,condition in evaluation['results'].items():
        rows=[d['conditions'][name] for d in documents];tokens=sum(r['tokens'] for r in rows)
        for metric,column in [('teacher_kl_nats','kl'),('dense_nll','teacher_nll'),('student_nll','student_nll'),('top1_agreement','top1_agreement')]:
            assert math.isclose(condition['overall'][metric],sum(r[column] for r in rows)/tokens,abs_tol=1e-12,rel_tol=1e-12)
        assert math.isclose(condition['overall']['student_perplexity'],math.exp(condition['overall']['student_nll']),abs_tol=1e-12)
        for domain,summary in condition['domains'].items():
            domain_rows=[r for r in rows if r['domain']==domain]
            assert summary['tokens']==sum(r['tokens'] for r in domain_rows)
            for metric,column in [('teacher_kl_nats','kl'),('dense_nll','teacher_nll'),('student_nll','student_nll'),('top1_agreement','top1_agreement')]:
                assert math.isclose(summary[metric],sum(r[column] for r in domain_rows)/summary['tokens'],abs_tol=1e-12,rel_tol=1e-12)
    for name,recovery in evaluation['recovered_advantage'].items():
        pct=name.split('-')[-1]
        for metric in ['student_perplexity','student_nll','teacher_kl_nats']:
            f=evaluation['results'][f'fixed-{pct}']['overall'][metric]
            p=evaluation['results'][name]['overall'][metric]
            o=evaluation['results'][f'oracle-{pct}']['overall'][metric]
            if f-o>1e-9:assert math.isclose(recovery[metric],(f-p)/(f-o),abs_tol=1e-12)
            else:assert recovery[metric] is None
    for p in source.glob('*.json'):shutil.copy2(p,out/p.name)
    # Paired uncertainty for the central comparison, conditional on this one
    # trained seed. Resample whole documents within each domain, never tokens.
    rng=np.random.default_rng(20261109)
    groups=[[i for i,d in enumerate(documents) if d['metadata']['domain']==domain]
            for domain in ['web','code','conversation']]
    draws=np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(2000)])
    comparison={}
    for pct in [50,40]:
        rows=np.array([[d['conditions'][f'binary-{pct}']['student_nll']-d['conditions'][f'hidden-{pct}']['student_nll'],
                        d['conditions'][f'binary-{pct}']['kl']-d['conditions'][f'hidden-{pct}']['kl'],
                        d['conditions'][f'binary-{pct}']['tokens']] for d in documents])
        totals=rows.sum(0);bootstrap=rows[draws].sum(1)
        comparison[str(pct)]={metric:{'binary_minus_hidden':float(totals[i]/totals[2]),
            'paired_ci95':np.quantile(bootstrap[:,i]/bootstrap[:,2],[.025,.975]).tolist()}
            for i,metric in enumerate(['nll_nats','teacher_kl_nats'])}
    comparison['interpretation']='Positive means binary has higher error. Paired, domain-stratified document bootstrap; one training seed, not training-seed uncertainty.'
    (out/'paired-predictor-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    report={'passed':True,'documents':len(documents),'conditions':len(evaluation['results']),
        'metrics_reconstructed':True,'domain_metrics_reconstructed':True,
        'checkpoint_hashes_match':True,'recovery_points_reconstructed':True}
    (out/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
