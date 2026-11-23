import json
from pathlib import Path
from scripts.mistral_full import update_selection


def test_plateau_runs_hardening_and_keeps_best_validation_checkpoint():
    c=json.loads(Path('configs/mistral-planner-full.json').read_text())
    s=dict(best_kl=None,best_tokens=None,stale_checks=0,stop_at=c['e2e_tokens'],hard_tail=False,curve=[])
    def observe(tokens,kl):
        s['pending_validation']=tokens
        update_selection(s,tokens,{'overall':{'teacher_kl_nats':kl,'relative_ppl_increase':.1,'routed_kept':.5}},c)
        assert 'pending_validation' not in s
    observe(5_000_000,.1)
    observe(10_000_000,.1001)
    assert not s['hard_tail']
    observe(15_000_000,.1002)
    assert s['hard_tail'] and s['stop_at']==15_500_000
    assert s['best_tokens']==5_000_000
    observe(15_500_000,.09)
    assert s['best_tokens']==15_500_000
    assert s['stop_at']==15_500_000


def test_improving_validation_does_not_stop_early():
    c=json.loads(Path('configs/mistral-planner-full.json').read_text())
    s=dict(best_kl=None,best_tokens=None,stale_checks=0,stop_at=c['e2e_tokens'],hard_tail=False,curve=[])
    for tokens,kl in [(5_000_000,.2),(10_000_000,.15),(15_000_000,.1),(20_000_000,.08)]:
        update_selection(s,tokens,{'overall':{'teacher_kl_nats':kl,'relative_ppl_increase':.1,'routed_kept':.4}},c)
    assert not s['hard_tail'] and s['stop_at']==20_000_000
    assert s['best_tokens']==20_000_000
