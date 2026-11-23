import math
import pytest
import torch
from pathpred.oracle_checks import oracle_mask, mask_to_gates, quality_rows, concentration


def test_binary_oracle_exact_counts_strong_priority_and_seed():
    norms = torch.tensor([[[1., 2., 3., 4.]] * 4] * 8)
    q = torch.full((4, 4), 2.)
    a = oracle_mask(norms, q, [1, 3], seed=3)
    assert a[:, :2].all()
    assert (a[:, 2:].sum(-1) == torch.tensor([1, 3])).all()
    assert not a[:, 2, :2].any() and a[:, 3, 2:].all()
    assert torch.equal(a, oracle_mask(norms, q, [1, 3], seed=3))
    assert not torch.equal(a, oracle_mask(norms, q, [1, 3], seed=4))
    raw = oracle_mask(norms, q, mode='threshold')
    assert raw[:, :2].all() and torch.equal(raw[:, 2:], norms[:, 2:] > q[2:])


def test_ranked_oracle_normalizes_each_unit_and_preserves_zero_threshold_priority():
    norms = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [.001, 90., 2.]]])
    q = torch.tensor([[1., 1., 1.], [1., 1., 1.], [0., 100., 1.]])
    m = oracle_mask(norms, q, [2], mode='ranked')
    assert m[0, 2].tolist() == [True, False, True]
    gates = mask_to_gates(m, query_heads=1)
    assert gates[(2, 'attention')].shape == (1, 1, 1)
    assert gates[(2, 'mlp')].shape == (1, 1, 2)


def test_quality_uses_next_token_full_vocab_and_ignores_last_logit():
    a = torch.tensor([[[2., 0., -2.], [0., 2., -2.], [99., -99., 0.]]])
    b = torch.zeros_like(a)
    ids = torch.tensor([[2, 0, 1]])
    r = quality_rows(a, b, ids, chunk=1)
    assert len(r['kl']) == 2
    assert torch.allclose(r['student_nll'], torch.full((2,), math.log(3)))
    expected = -a[0, :2].log_softmax(-1)[torch.arange(2), torch.tensor([0, 1])]
    torch.testing.assert_close(r['teacher_nll'], expected)
    assert (r['kl'] > 0).all()
    same = quality_rows(a, a, ids)
    assert torch.equal(same['kl'], torch.zeros(2))
    assert same['top1_agreement'].eq(1).all()


def test_grouping_preserves_energy_but_reduces_concentration_resolution():
    x = torch.tensor([[9., 0., 1., 0.], [0., 0., 0., 0.]])
    neuron = concentration(x)
    group = concentration(x.reshape(2, 2, 2).sum(-1))
    assert neuron['fraction_for_50pct_energy'][0] == .25
    assert group['fraction_for_50pct_energy'][0] == .5
    assert neuron['fraction_for_95pct_energy'][0] == .5
    assert group['fraction_for_95pct_energy'][0] == 1
    assert neuron['fraction_for_95pct_energy'][1] == 0
    with pytest.raises(ValueError):
        concentration(-torch.ones(1, 4))


def test_quality_sampling_uses_distinct_documents_and_available_inventory():
    from scripts.oracle_quality import select_documents
    rows=[]
    for domain,count in [('web',25),('code',15),('conversation',24)]:
        for i in range(count):
            for window in range(2):
                rows.append({'split':'test','domain':domain,'length':512,
                    'document_sha256':domain+str(i),'sequence_id':len(rows)})
    selected=select_documents(rows,'test',24,{'seed':41,'minimum_sequence_length':128})
    assert len(selected)==63
    assert len({r['document_sha256'] for r in selected})==63
    assert selected==select_documents(rows,'test',24,{'seed':41,'minimum_sequence_length':128})
