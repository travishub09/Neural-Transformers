"""Calibrate the effective centroid intersection without touching evaluation data."""
import torch
from .neuroprefetcher import prune_aggregate


def banks_at_count(aggregates,count,units=6144):
    return {key:prune_aggregate(value,count/units) for key,value in aggregates.items()}


def intersection_retention(gate_ids,up_ids,banks,prefix=2):
    total=0.;count=0
    for index in range(gate_ids.shape[-1]):
        layer=prefix+index
        gate=banks[f'gate_{layer}'].float();up=banks[f'up_{layer}'].float()
        # KxK overlaps avoid materializing tokens x layers x neurons.
        overlaps=gate@up.T
        selected=overlaps[gate_ids[...,index],up_ids[...,index]]
        total+=float(selected.double().sum());count+=selected.numel()*gate.shape[-1]
    return total/count


def calibrate_count(measure,target,units=6144,tolerance=.0005):
    """Integer centroid density; explicitly flag an unattainable target."""
    history={}
    def value(count):
        if count not in history:history[count]=float(measure(count))
        return history[count]
    lo,hi=0,units
    assert value(lo)<=target<=value(hi)
    while hi-lo>1:
        mid=(lo+hi)//2
        if value(mid)<target:lo=mid
        else:hi=mid
    best=min((lo,hi),key=lambda k:(abs(value(k)-target),value(k)>target))
    actual=value(best)
    return {'count_per_centroid':best,'target_intersection_retention':target,
        'validation_retention':actual,'absolute_error':abs(actual-target),
        'matched_budget':abs(actual-target)<=tolerance,'tolerance':tolerance,
        'measurements':{str(k):v for k,v in sorted(history.items())},
        'calibration_variable':'common number of retained members per gate/up centroid'}
