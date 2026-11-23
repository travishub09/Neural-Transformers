import numpy as np
import pytest
import torch
from pathpred.phase0 import (coverage_labels, budget_plan, mask_for_budget, select_units,
                            weighted_loss, metric_rows, summarize_rows, paired_document_bootstrap)


def costs():
    return {'layers': 4, 'scenarios': {'1024': {'per_layer_unit_flops': [10] * 4,
        'all_layers_gated_flops': 160, 'mandatory_kv_projection_flops': 10,
        'mandatory_lm_head_flops': 20, 'dense_accounted_flops': 190}}}


def test_coverage_targets_are_minimal_per_family_with_zero_and_ties():
    x = torch.tensor([[[8., 1., 1., 0., 3., 1.]], [[0., 0., 0., 0., 0., 0.]]])
    labels, shares = coverage_labels(x, .8, ((0, 4), (4, 6)))
    assert labels[0, 0].tolist() == [True, False, False, False, True, True]
    assert not labels[1].any()
    assert shares[0].sum() == 1 and shares[1].sum() == 0
    tie, _ = coverage_labels(torch.ones(1, 1, 4), .5, ((0, 4),))
    assert tie.tolist() == [[[True, True, False, False]]]


def test_matched_budget_pays_for_router_and_dense_warmup():
    a = budget_plan(costs(), .75, 10, 10, lookahead=1)
    b = budget_plan(costs(), .75, 30, 10, lookahead=1)
    assert a['total_matmul_cap'] == b['total_matmul_cap'] == 160
    assert sum(a['counts']) == 8 and sum(b['counts']) == 6
    assert a['accounted_matmul_used'] == b['accounted_matmul_used'] == 160
    with pytest.raises(ValueError, match='infeasible'):
        budget_plan(costs(), .1, 10, 10, lookahead=2)


def test_random_masks_exact_counts_warmup_reproducibility():
    a = mask_for_budget(9, 5, 4, [1, 2, 3], 23)
    assert a[:, :2].all() and a[:, 2:].sum(-1).tolist() == [[1, 2, 3]] * 9
    assert torch.equal(a, mask_for_budget(9, 5, 4, [1, 2, 3], 23))
    assert not torch.equal(a, mask_for_budget(9, 5, 4, [1, 2, 3], 24))


def test_selection_metrics_and_loss_have_known_values():
    scores = torch.tensor([[[4., 3., 2., 1.], [1., 2., 3., 4.]]], requires_grad=True)
    selected = select_units(scores, [1, 2])
    assert selected.tolist() == [[[True, False, False, False], [False, False, True, True]]]
    labels = torch.tensor([[[True, True, False, False], [False, False, True, True]]])
    shares = torch.full_like(scores, .25)
    rows = metric_rows(selected, labels, shares, labels, ((0, 2), (2, 4)))
    summary = summarize_rows({k: v.numpy() for k, v in rows.items()})
    assert summary['recall'] == .75 and summary['precision'] == 1
    assert summary['norm_mass_fraction'] == .375
    weighted_loss(scores, labels, families=((0, 2), (2, 4))).backward()
    assert torch.isfinite(scores.grad).all()


def test_bootstrap_clusters_documents_and_preserves_pairing():
    a = np.array([[2., 2., 2., 2.], [2., 2., 2., 2.]])
    b = a - 1
    x = paired_document_bootstrap(a, b, np.full(4, 4.), [0, 0, 1, 2], [0, 0, 0, 1], repetitions=100)
    assert x['difference'] == .25 and x['ci95'] == [.25, .25] and x['documents'] == 3
    z = paired_document_bootstrap(a, a, np.full(4, 4.), [0, 0, 1, 2], [0, 0, 0, 1], repetitions=100)
    assert z['ci95'] == [0., 0.]


def test_pair_inputs_hide_both_tokens_and_keep_missing_distinct_from_zero():
    from pathpred.phase0_training import PhaseData
    data = PhaseData.__new__(PhaseData)
    data.device, data.layers = 'cpu', 3
    data.t = {'previous_sample_indices': torch.tensor([-1, 0]),
              'binary': torch.tensor([[[1, 0]] * 3, [[0, 1]] * 3], dtype=torch.int8)}
    data.hidden = torch.randn(2, 3, 4)
    mask = torch.tensor([[[False, True]] * 3, [[True, False]] * 3])
    x = data.inputs(torch.tensor([1]), mask, 'binary')
    assert x['states'].tolist() == [[[0, -1]] * 3]
    assert x['previous'].tolist() == [[[-1, 0]] * 3]
    altered = data.t['binary'].clone()
    altered[~mask] = 1-altered[~mask]
    data.t['binary'] = altered
    y = data.inputs(torch.tensor([1]), mask, 'binary')
    assert torch.equal(x['states'], y['states']) and torch.equal(x['previous'], y['previous'])
    h = data.inputs(torch.tensor([1]), mask, 'hidden')
    assert 'states' not in h and torch.equal(h['previous_hidden'], data.hidden[:1])
    with pytest.raises(ValueError, match='adjacent'):
        data.inputs(torch.tensor([0]), mask, 'binary')


def test_all_five_signals_train_and_reload_on_known_targets(tmp_path):
    from pathpred.path_model import PathConfig, PathModel
    from safetensors.torch import save_file, load_file
    torch.set_num_threads(2)
    for signal in ['decisions', 'binary', 'coarse', 'magnitudes', 'hidden']:
        torch.manual_seed(4)
        config = PathConfig(llm_layers=4, units=4, width=8, depth=1, heads=2,
                            hidden_size=6, hidden_summary=3, signal=signal)
        model = PathModel(config)
        codes = torch.randint(-1, 2, (8, 4, 4), dtype=torch.int8)
        valid = torch.ones(8, 4, dtype=torch.bool)
        if signal == 'hidden':
            x = {'hidden': torch.randn(8, 4, 6), 'previous_hidden': torch.randn(8, 4, 6)}
        else:
            if signal == 'magnitudes':
                codes = codes.float()
            x = {'states': codes, 'previous': codes.flip(0)}
        x['previous_valid'] = valid
        labels = torch.zeros(8, 2, 4, dtype=torch.bool)
        labels[..., :2] = True
        opt = torch.optim.AdamW(model.parameters(), lr=.03)
        initial = float(weighted_loss(model(**x), labels, families=((0, 2), (2, 4))).detach())
        for _ in range(15):
            opt.zero_grad()
            loss = weighted_loss(model(**x), labels, families=((0, 2), (2, 4)))
            loss.backward(); opt.step()
        expected = model(**x).detach()
        assert float(weighted_loss(expected, labels, families=((0, 2), (2, 4)))) < initial / 2
        path = tmp_path / (signal+'.safetensors')
        save_file(model.state_dict(), str(path))
        restored = PathModel(config)
        restored.load_state_dict(load_file(str(path)))
        torch.testing.assert_close(restored(**x), expected, rtol=0, atol=0)


def test_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    from pathpred.path_model import PathConfig
    import pathpred.phase0_training as training
    torch.set_num_threads(2)
    torch.manual_seed(91)
    data = training.PhaseData.__new__(training.PhaseData)
    data.n, data.layers, data.units, data.device = 12, 4, 64, 'cpu'
    data.t = {'binary': torch.randint(0, 2, (12, 4, 64), dtype=torch.int8),
              'labels': torch.rand(12, 4, 64) > .4,
              'shares': torch.full((12, 4, 64), 1/64),
              'previous_sample_indices': torch.tensor([-1, 0, -1, 2, -1, 4, -1, 6, -1, 8, -1, 10])}
    data.ids = {'train': torch.tensor([1, 3, 5, 7]), 'validation': torch.tensor([9, 11])}
    cfg = json.loads(Path('configs/phase0.json').read_text())
    cfg.update(epochs=3, minimum_epochs=3, batch_size=2)
    monkeypatch.setattr(training, 'model_config', lambda signal: PathConfig(
        llm_layers=4, units=64, width=8, depth=1, heads=2, signal=signal))
    arguments = (data, 'binary', .25, 101, cfg, {'counts': [10, 10]}, [10, 10])
    full = training.train_one(*arguments, tmp_path/'full', {'test': True}, float('inf'))
    original_save = training.save_torch_atomic
    def save_then_interrupt(path, content):
        original_save(path, content)
        if content['next_epoch'] == 1:
            raise RuntimeError('simulated interruption')
    monkeypatch.setattr(training, 'save_torch_atomic', save_then_interrupt)
    with pytest.raises(RuntimeError, match='interruption'):
        training.train_one(*arguments, tmp_path/'resume', {'test': True}, float('inf'))
    monkeypatch.setattr(training, 'save_torch_atomic', original_save)
    resumed = training.train_one(*arguments, tmp_path/'resume', {'test': True}, float('inf'))
    assert full['best'] == resumed['best']
    left = load_file(str(tmp_path/'full'/full['run']/'best.safetensors'))
    right = load_file(str(tmp_path/'resume'/resumed['run']/'best.safetensors'))
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_normalized_skill_random_oracle_and_negative_values():
    from pathpred.phase0 import normalized_skill_bootstrap
    lower=np.array([1.,1.,1.,1.]);upper=np.array([3.,3.,3.,3.])
    kwargs={'documents':[0,0,1,2],'domains':[0,0,0,1],'repetitions':100}
    for fraction in [0.,1.,-.5,.2]:
        hits=np.stack([lower+fraction*(upper-lower)]*3)
        result=normalized_skill_bootstrap(hits,lower,upper,**kwargs)
        assert result['value']==pytest.approx(fraction)
        assert result['ci95']==pytest.approx([fraction,fraction])
    undefined=normalized_skill_bootstrap(np.stack([upper]*3),upper,upper,**kwargs)
    assert undefined['value'] is None and undefined['ci95'] is None
