import pytest
from scripts.mistral_long_prompt import crop, ranked_indices, cap_retention, matched_cap


def test_both_prompt_lengths_score_identical_targets_with_nested_context():
    seq=list(range(1536))
    short=crop(seq,512);long=crop(seq,1024)
    assert len(short)==1024 and len(long)==1536
    assert short[512:]==long[1024:]==list(range(1024,1536))
    assert short[:512]==long[512:1024]
    with pytest.raises(ValueError):crop(seq[:1024],512)


def test_block_selection_is_reproducible_and_fails_if_corpus_too_short():
    a=ranked_indices(200,64,'test',123)
    assert len(a)==len(set(a))==64
    assert a==ranked_indices(200,64,'test',123)
    assert a!=ranked_indices(200,64,'validation',123)
    with pytest.raises(ValueError):ranked_indices(20,64,'test',123)


def test_capped_histogram_matches_explicit_per_layer_clipping():
    counts=[2000,6000,7000,7168,7168,7168]
    hist={str(n):counts.count(n) for n in set(counts)}
    for cap in [1,5734,6881,7168]:
        expected=sum(min(n,cap) for n in counts)/len(counts)/14336
        assert cap_retention(hist,cap)==pytest.approx(expected)
    target=.35;cap=matched_cap(hist,target)
    assert abs(cap_retention(hist,cap)-target)<=1/14336
    assert abs(cap_retention(hist,cap)-target)<=abs(cap_retention(hist,cap+1)-target)


def test_histogram_cannot_invent_more_than_native_retention():
    hist={2000:10}
    cap=matched_cap(hist,.48)
    assert cap_retention(hist,cap)<.48-.001
    with pytest.raises(ValueError):cap_retention({},10)
