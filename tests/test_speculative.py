import torch
from transformers import MistralConfig, MistralForCausalLM
from contextlib import contextmanager
from pathpred.speculative import dense_greedy, speculative_greedy
from scripts.mistral_comparison_core import Policy,score_batch


class Control:
    @contextmanager
    def use_gates(self, *args): yield


def policy(seed=7):
    torch.manual_seed(seed)
    c = MistralConfig(vocab_size=31, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=256,
                      sliding_window=None, attn_implementation='eager')
    return Policy(MistralForCausalLM(c).eval(), Control(), 'dense')


def test_full_acceptance_and_output_caps():
    p = policy(); prompt=[1, 3, 7]
    reference=dense_greedy(p,prompt,13)['tokens']
    for block in [1,4,8]:
        out=speculative_greedy(p,prompt,13,block)
        assert out['tokens']==reference
        assert all(s['accepted']==s['proposed'] for s in out['steps'])


def test_rejection_rolls_back_cache_and_history():
    p=policy(); prompt=[1,3,7]; reference=dense_greedy(p,prompt,17)['tokens']
    original=p.step
    # Force bad proposals and a history value tied to the current KV length.
    def bad_step(ids, valid):
        if p.previous is not None:
            assert p.previous == p.cache.get_seq_length()
        z=original(ids,valid)
        p.previous=p.cache.get_seq_length()
        return z.roll(1,-1)
    p.step=bad_step
    out=speculative_greedy(p,prompt,17,4)
    assert out['tokens']==reference
    assert any(s['accepted']<s['proposed'] for s in out['steps'])


def test_eos_matches_dense():
    p=policy(); prompt=[2,4,6]
    eos=dense_greedy(p,prompt,1)['tokens'][0]
    assert speculative_greedy(p,prompt,17,4,eos)['tokens']==[eos]


def test_position_logging_reconstructs_document_totals():
    p=policy();captured=[]
    def capture(t,vals,live,ret):captured.append(vals.double()*live[:,None])
    z=score_batch(p,[[1,2,3,4,5],[2,3,4,5]],[2,2],position_callback=capture)
    torch.testing.assert_close(torch.stack(captured).sum(0),z['sums'])
