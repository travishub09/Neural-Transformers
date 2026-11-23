import json
import pytest
from scripts.mistral_hellaswag_matched import select_indices, checkpoint_identity, paired
from pathpred.collection import file_sha256


def test_calibration_selection_is_unique_deterministic_and_not_prefix():
    selected=select_indices(10000)
    assert len(selected)==len(set(selected))==1024
    assert selected==select_indices(10000) and selected!=list(range(1024))


@pytest.mark.parametrize('budget,family,config',[(50,'mistral-phased-v1','mistral-planner-phased.json'),
    (30,'mistral-lower-v1','mistral-lower-budgets.json'),(20,'mistral-lower-v1','mistral-lower-budgets.json')])
def test_checkpoint_identity_uses_manifest_selected_weights(tmp_path,monkeypatch,budget,family,config):
    monkeypatch.chdir(tmp_path)
    root=tmp_path/f'artifacts/{family}/budget-{budget}';root.mkdir(parents=True)
    (root/'completion.json').write_text(json.dumps({'tokens':123}))
    (root/'selected.safetensors').write_bytes(b'weights')
    (root/'checkpoint-00000123.json').write_text(json.dumps({'file':{'file':'selected.safetensors',
        'sha256':file_sha256(root/'selected.safetensors')}}))
    (root/'check-00000123').mkdir();(root/'check-00000123/validation.json').write_text('{}')
    (tmp_path/'artifacts/mistral-pilot-v1').mkdir()
    (tmp_path/'artifacts/mistral-pilot-v1/thresholds.safetensors').write_bytes(b'q')
    (tmp_path/'configs').mkdir();(tmp_path/'configs'/config).write_text('{}')
    identity=checkpoint_identity(budget)
    assert f'artifacts/{family}/budget-{budget}/selected.safetensors' in identity
    (root/'selected.safetensors').write_bytes(b'changed')
    with pytest.raises(AssertionError):checkpoint_identity(budget)


def test_paired_report_aligns_ids_and_rejects_different_labels():
    a=[dict(id=i,gold=i%4,accuracy_normalized=1) for i in range(10042)]
    b=[dict(id=i,gold=i%4,accuracy_normalized=0) for i in reversed(range(10042))]
    assert paired(a,b)['gap_percentage_points']==100
    b[0]['gold']=99
    with pytest.raises(AssertionError):paired(a,b)


def test_calibration_fingerprints_code_before_loading_model(tmp_path,monkeypatch):
    from scripts import mistral_hellaswag_matched as m
    monkeypatch.setattr(m,'OUT',tmp_path);monkeypatch.setattr(m,'CAL',tmp_path)
    (tmp_path/'examples.jsonl').write_text('{}\n')
    monkeypatch.setattr(m,'checkpoint_identity',lambda budget:{})
    def stop(*args):raise RuntimeError('model-load-sentinel')
    monkeypatch.setattr(m,'load',stop)
    with pytest.raises(RuntimeError,match='model-load-sentinel'):m.calibrate(50)
    assert len(json.loads((tmp_path/'calibration-50-identity.json').read_text())['code'])==4


@pytest.mark.parametrize('budget',[50,30,20])
def test_custom_evaluation_fingerprints_before_loading_model(tmp_path,monkeypatch,budget):
    import sys
    from scripts import mistral_hellaswag as m
    monkeypatch.chdir(tmp_path)
    (tmp_path/'scripts').mkdir()
    for name in ['mistral_hellaswag.py','mistral_comparison_core.py','mistral_comparisons.py']:
        (tmp_path/'scripts'/name).write_text('# fixture')
    data=tmp_path/'data/hellaswag-mistral-v1';data.mkdir(parents=True)
    (data/'examples.jsonl').write_text('{}\n')
    calibration=tmp_path/'calibration.json';calibration.write_text(json.dumps({'checkpoint_budget':budget}))
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['eval','--kind','planner','--budget',str(budget),'--limit','1','--calibration',str(calibration),'--output-dir',str(out)])
    def stop(*args):raise RuntimeError('model-load-sentinel')
    monkeypatch.setattr(m,'load',stop)
    with pytest.raises(RuntimeError,match='model-load-sentinel'):m.main()
    assert len(json.loads((out/f'planner-{budget}-identity.json').read_text())['code'])==3


def test_lower_report_references_do_not_invent_lower_budget_baselines():
    from scripts.mistral_hellaswag_matched import references_for
    refs=references_for(20)
    assert 'griffin-40' in refs and 'planner-48-matched' in refs and 'dense-H200' in refs
    assert 'griffin-20' not in refs and 'neuro-30' not in refs
