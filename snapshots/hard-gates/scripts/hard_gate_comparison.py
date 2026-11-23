"""A single-variable gate-noise ablation, reusing the completed matched control."""
import argparse
import json
from pathlib import Path
import signal
import shutil
import torch
from pathpred.collection import atomic_json,file_sha256
from scripts import planner_experiment as pe
from scripts.report_matched_e2e import audit,paired_difference

ROOT=Path('artifacts/hard-gate-comparison-v1')
REFERENCE=Path('/workspace/pathpred-planner/artifacts/planner-v1')
CONFIG=Path('configs/hard-gate-comparison.json')


def prepare(c):
    ROOT.mkdir(parents=True,exist_ok=True)
    source=REFERENCE/'autoregressive'
    checkpoint=json.loads((source/'scheduled/checkpoint-01000000.json').read_text())
    assert checkpoint['file']['sha256']=='a474ce2d3c67fab8c8a628cb0ca62244c714ef793cb348a5054f86ed532957d6'
    assert file_sha256(source/'scheduled'/checkpoint['file']['file'])==checkpoint['file']['sha256']
    for name,digest in checkpoint['fingerprint']['files'].items():
        if name.startswith(('pathpred/','data/','artifacts/neuron-guessing-v1/')):
            assert file_sha256(Path(name))==digest,name
    original=json.loads(Path('configs/planner-pilot.json').read_text())
    unchanged=['seed','e2e_tokens_per_budget','sequence_length','width','depth','heads',
        'quality_learning_rate','weight_decay','warmup_tokens','beta','epsilon','alpha_initial',
        'dual_learning_rate','maximum_alpha','hard_fraction','temperatures','quality_checkpoint_tokens',
        'teacher_head_chunk_size','quality_corpus','eval_batch_size','validation_documents_per_domain',
        'calibration_tolerance','calibration_max_evaluations']
    for key in unchanged:assert c[key]==original[key],key
    dest=ROOT/'autoregressive/scheduled';dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.symlink_to((source/'scheduled').resolve(),target_is_directory=True)
    control=ROOT/'control';control.mkdir(exist_ok=True)
    for name in ['evaluation.json','evaluation-documents.json','threshold.json','initial-offsets.json',
                 'training.json','completion.json','checkpoint-02000000.json']:
        shutil.copyfile(source/'budget-50'/name,control/name)
    pe.ROOT=ROOT
    pe.ordered_data(c,quality=True)
    current=json.loads((ROOT/'quality-data-audit.json').read_text())
    old=json.loads((REFERENCE/'quality-data-audit.json').read_text())
    assert current['order_sha256']==old['order_sha256']
    atomic_json(ROOT/'comparison-protocol.json',{'model':'Qwen3-1.7B-Base, frozen, original clustered neuron order',
        'predictor':'Original upfront autoregressive planner with previous-token path',
        'source_checkpoint_sha256':checkpoint['file']['sha256'],'e2e_tokens_each':c['e2e_tokens_per_budget'],
        'control':'Existing 2M run: noisy binary gates for first 80%, deterministic for final 20%',
        'treatment':'Deterministic binary gates throughout new E2E training; no gate sampling',
        'unchanged_settings_verified':unchanged,'training_order_sha256':current['order_sha256'],
        'evaluation':'Same validation calibration and same 63 test documents; actual retention reported',
        'one_seed':True,'warm_start_contains_prior_copy_and_scheduled_training':True,
        'not_a_claim_of_deterministic_floating_point_arithmetic':True,
        'control_retrained':False,'physical_sparse_kernels':False})


def report():
    treatment=ROOT/'autoregressive/budget-50'
    a,arows=audit(treatment/'evaluation.json');b,brows=audit(ROOT/'control/evaluation.json')
    ao=json.loads((treatment/'initial-offsets.json').read_text())
    bo=json.loads((ROOT/'control/initial-offsets.json').read_text())
    delta=max(abs(x-y) for x,y in zip(ao,bo));assert len(ao)==len(bo)==26
    assert delta<1e-5,('Initial offsets differ',delta)
    at=json.loads((treatment/'threshold.json').read_text());bt=json.loads((ROOT/'control/threshold.json').read_text())
    assert at['identity']['validation_documents']==bt['identity']['validation_documents']
    for t in [at,bt]:assert abs(t['validation_retention']-.5)<=t['tolerance']
    comparison=paired_difference(arows,brows)
    result={'hard_only':a,'noisy_then_hard_control':b,'hard_minus_control':comparison,
        'initial_offsets_max_difference':delta,'protocol':json.loads((ROOT/'comparison-protocol.json').read_text())}
    atomic_json(ROOT/'results.json',result)
    out=Path('results/hard-gate-comparison-v1');out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'results.json',result)
    for src,name in [(treatment/'evaluation-documents.json','hard-documents.json'),
                     (ROOT/'control/evaluation-documents.json','control-documents.json'),
                     (treatment/'threshold.json','hard-calibration.json'),
                     (ROOT/'control/threshold.json','control-calibration.json')]:shutil.copyfile(src,out/name)
    lines=['# Qwen3-1.7B deterministic-gate comparison','',
        'Same warm start and 2M E2E tokens; only training gate-noise policy differs. One seed. '
        'Deterministic exact-history evaluation throughout. Budgets calibrated on validation; test retention can differ.','',
        '| Training policy | Test MLP kept | PPL increase | KL | Top-token agreement |',
        '|---|---:|---:|---:|---:|']
    for label,r in [('Deterministic throughout',a),('Noisy first 80%, deterministic last 20%',b)]:
        o=r['overall'];lines.append(f"| {label} | {o['routed_kept']:.3%} | {o['relative_ppl_increase']:+.3%} | {o['teacher_kl_nats']:.5f} | {o['top1_agreement']:.3%} |")
    lines+=['',f"Paired PPL-increase difference (hard minus control): {comparison['ppl_increase_difference_pp']:+.3f} percentage points; 95% document-bootstrap interval {comparison['ppl_difference_ci95_pp']}. Negative favors hard-only.",
        '', 'Document-bootstrap intervals do not measure training-seed uncertainty. No physical inference speedup is measured.',
        '', 'Full domain metrics, actual retention differences, and document-level rows are in the adjacent JSON files.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','report']);args=parser.parse_args()
    c=json.loads(CONFIG.read_text());pe.ROOT=ROOT
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    signal.signal(signal.SIGUSR1,pe.request_pause);signal.signal(signal.SIGTERM,pe.request_pause)
    if args.stage=='prepare':prepare(c)
    elif args.stage=='train':pe.train(c,CONFIG,'autoregressive','e2e',50,gate_policy='hard_only')
    else:report()


if __name__=='__main__':main()
