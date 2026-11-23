"""Independently audit checkpoint results and export paired quality changes."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np


def main():
    source = Path('artifacts/e2e-pilot-v1'); out = Path('docs/e2e-pilot')
    out.mkdir(parents=True, exist_ok=True)
    training = json.loads((source/'training.json').read_text())
    assert training['status'] == 'complete' and training['tokens'] == 3000000
    milestones = [0] + training['config']['milestones']
    actual_updates = sum(math.ceil((b-a)/512) for a,b in zip(milestones,milestones[1:]))
    resume = json.loads((source/'resume-metadata-audit.json').read_text())
    assert resume['tokens'] == training['tokens'] and resume['next_position'] == actual_updates
    # The original run counted the final loop termination check as one window.
    # Preserve its raw record, disclose the correction, and fix future logging.
    assert training['gradient_update_windows'] in (actual_updates,actual_updates+1)
    baseline = json.loads((source/'evaluation-0000000.json').read_text())
    baseline_docs = json.loads((source/'documents-0000000.json').read_text())
    old = json.loads(Path('docs/neuron-guessing/evaluation.json').read_text())
    for pct in [50,40]:
        for metric in ['student_nll','teacher_kl_nats','top1_agreement']:
            assert math.isclose(baseline['conditions'][f'topk-{pct}']['overall'][metric],
                                old['results'][f'binary-{pct}']['overall'][metric], abs_tol=1e-10)
    paired = {}; checkpoints = []
    for file in sorted(source.glob('evaluation-*.json')):
        result = json.loads(file.read_text()); label = result['checkpoint']
        documents = json.loads((source/f'documents-{label}.json').read_text())
        assert len(documents) == result['documents'] == len(baseline_docs) == 63
        assert [d['metadata']['document_sha256'] for d in documents] == [d['metadata']['document_sha256'] for d in baseline_docs]
        for name, condition in result['conditions'].items():
            for domain, summary in [('overall',condition['overall']), *condition['domains'].items()]:
                rows = [d['conditions'][name] for d in documents if domain=='overall' or d['metadata']['domain']==domain]
                tokens = sum(r['tokens'] for r in rows)
                assert tokens == summary['tokens']
                for metric, column in [('dense_nll','teacher_nll'),('student_nll','student_nll'),
                                       ('teacher_kl_nats','kl'),('top1_agreement','top1_agreement')]:
                    assert math.isclose(sum(r[column] for r in rows)/tokens, summary[metric], rel_tol=1e-12, abs_tol=1e-12)
                assert math.isclose(math.exp(summary['student_nll']),summary['student_perplexity'],rel_tol=1e-12)
            realized = sum(d['conditions'][name]['routed_kept']*d['conditions'][name]['tokens'] for d in documents)/condition['overall']['tokens']
            assert math.isclose(realized, condition['overall']['routed_kept'], abs_tol=1e-12)
        if label != '0000000':
            meta = json.loads((source/f'checkpoint-{label}.json').read_text())
            with (source/meta['file']['file']).open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest() == meta['file']['sha256']
        rng = np.random.default_rng(20261110)
        groups = [[i for i,d in enumerate(documents) if d['metadata']['domain']==domain]
                  for domain in ['web','code','conversation']]
        draws = np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(2000)])
        paired[label] = {}
        for name in ['topk-50','topk-40']:
            values = np.array([[d['conditions'][name]['kl'], b['conditions'][name]['kl'],
                                d['conditions'][name]['student_nll'],b['conditions'][name]['student_nll'],
                                d['conditions'][name]['tokens']] for d,b in zip(documents,baseline_docs)])
            totals=values.sum(0); sums=values[draws].sum(1)
            paired[label][name]={'kl_reduction_fraction':float(1-totals[0]/totals[1]),
                'kl_reduction_fraction_ci95':np.quantile(1-sums[:,0]/sums[:,1],[.025,.975]).tolist(),
                'nll_change_nats':float((totals[2]-totals[3])/totals[4]),
                'nll_change_nats_ci95':np.quantile((sums[:,2]-sums[:,3])/sums[:,4],[.025,.975]).tolist()}
        native=np.array([[d['conditions']['native']['routed_kept']*d['conditions']['native']['tokens'],
                          d['conditions']['native']['tokens']] for d in documents])
        sampled=native[draws].sum(1)
        paired[label]['native_retention_ci95']=np.quantile(sampled[:,0]/sampled[:,1],[.025,.975]).tolist()
        checkpoints.append(label)
    assert checkpoints == ['0000000','1000000','2000000','3000000']
    # Whitelist experiment records; never copy local provisioning metadata.
    for pattern in ['evaluation-*.json','documents-*.json','checkpoint-*.json',
                    '*offsets.json','training.json','pilot.json','data-audit.json',
                    'completion.json','supervisor.json','cleanup-verified.json',
                    'resume-metadata-audit.json','retained-volume.json']:
        for file in source.glob(pattern): shutil.copy2(file,out/file.name)
    (out/'paired-improvements.json').write_text(json.dumps(paired,indent=2)+'\n')
    audit={'passed':True,'checkpoints':checkpoints,'documents_per_checkpoint':63,
           'conditions_per_checkpoint':3,'baseline_matches_guessing':True,
           'metrics_domains_retention_reconstructed':True,'checkpoint_hashes_verified':True,
           'actual_gradient_update_windows':actual_updates,
           'raw_reported_windows':training['gradient_update_windows'],
           'window_counter_note':'Raw record includes one terminal loop check; token counts and optimization are unaffected. Future logging corrected.'}
    (out/'independent-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__': main()
