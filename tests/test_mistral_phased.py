import copy
import json
import math
from pathlib import Path
import pytest
import torch
from scripts.mistral_phased import migration_state,schedule,update_selection


def config():return json.loads(Path('configs/mistral-planner-phased.json').read_text())


def checkpoint(tokens):
    return {'model':{'weight':torch.tensor([1.,2.])},'optimizer':{'state':{'moment':torch.tensor([.2,.4])}},
        'tokens':tokens,'position':42972,'elapsed':123.,'cpu_rng':torch.get_rng_state(),
        'cuda_rng':torch.tensor([3]),'offsets':torch.tensor([.1,.2]),
        'dual':{'target':.5,'alpha':.7,'learning_rate':.05,'maximum':100},
        'selection':{'curve':[],'stop_at':20_000_000,'pending_validation':tokens}}


def test_transfer_preserves_training_state_and_pending_calibration():
    saved=checkpoint(10_000_000);before=copy.deepcopy(saved);new=migration_state(saved,config(),50)
    assert new['position']==before['position'] and new['tokens']==10_000_000
    assert new['model'] is saved['model'] and new['optimizer'] is saved['optimizer']
    assert torch.equal(new['cpu_rng'],before['cpu_rng'])
    assert new['selection']['pending_validation']==10_000_000
    assert new['selection']['stop_at']==25_000_000
    assert saved['selection']==before['selection']


def test_phase40_inherits_phase50_without_reusing_its_data():
    saved=checkpoint(25_000_000);new=migration_state(saved,config(),40)
    assert new['model'] is saved['model'] and new['optimizer'] is saved['optimizer']
    assert new['position']==saved['position'] and new['tokens']==0
    assert new['dual']['target']==.4 and new['selection']['stop_at']==15_000_000
    assert 'pending_validation' not in new['selection']
    with pytest.raises(AssertionError):migration_state(checkpoint(20_000_000),config(),40)


def test_phase50_schedule_is_continuous_at_transfer_and_hardens():
    c=config();origin=11_000_000
    lr,temp,mode=schedule(c,50,origin,origin)
    assert lr==pytest.approx(c['quality_learning_rate']*(.1+.45*(1+math.cos(math.pi*origin/20_000_000))))
    assert temp==pytest.approx(10**(-origin/16_000_000)) and mode=='gumbel'
    assert schedule(c,50,20_000_000,origin)[2]=='hard'
    assert schedule(c,50,25_000_000,origin)[0]==pytest.approx(.1*c['quality_learning_rate'])


@pytest.mark.parametrize('improvement,expected',[(.005,15_000_000),(.0001,10_000_000)])
def test_phase40_extension_uses_validation_and_is_hard_by_10M(improvement,expected):
    c=config();s={'curve':[{'tokens':5_000_000,'validation_kl':.2}],'stop_at':15_000_000}
    update_selection(s,10_000_000,{'overall':{'teacher_kl_nats':.2-improvement,
        'relative_ppl_increase':.1,'routed_kept':.4}},c,40)
    assert s['stop_at']==expected
    assert schedule(c,40,10_000_000,0)[2]=='hard'


def test_phase50_does_not_early_stop():
    c=config();s={'curve':[{'tokens':5_000_000,'validation_kl':.1}],'stop_at':25_000_000}
    update_selection(s,10_000_000,{'overall':{'teacher_kl_nats':.2,'relative_ppl_increase':.1,'routed_kept':.5}},c,50)
    assert s['stop_at']==25_000_000
