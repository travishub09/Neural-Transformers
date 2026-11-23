"""Batch-one PyTorch gather prototype. No inference speed claim without timing.

Gather gate/up rows and down columns before matmul. Preserve down_proj prehooks
so the existing planner observes only computed activations. CPU mode counts
actual tensor payload copied to CUDA, NOT host DRAM bus transactions.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F


class TransferMeter:
    def __init__(self): self.bytes=0; self.matrices=0
    def reset(self): self.bytes=0; self.matrices=0
    def move(self,weight,device):
        self.bytes+=weight.numel()*weight.element_size();self.matrices+=1
        return weight.to(device=device,non_blocking=False)


@contextmanager
def gathered_mlp(model,controller,*,cpu_weights=None,meter=None):
    originals=[]
    if cpu_weights is not None and meter is None:raise ValueError('CPU transfers require metering')
    for layer_id,layer in enumerate(model.model.layers):
        mlp=layer.mlp;down=mlp.down_proj
        original_mlp,original_down=mlp.forward,down.forward
        state={}
        def weights(lid,kind,weight,index,dim,device):
            if cpu_weights is None:
                return weight if index is None else weight.index_select(dim,index)
            host=cpu_weights[lid][kind]
            if kind=='down' and cpu_weights[lid].get('down_transposed',False):
                selected=host if index is None else host.index_select(0,index.cpu())
                return meter.move(selected,device).t()
            selected=host if index is None else host.index_select(dim,index.cpu())
            return meter.move(selected,device)
        def forward(x,lid=layer_id,m=mlp,orig=original_mlp,st=state):
            if x.shape[:2]!=(1,1):return orig(x)
            gate=controller.state.gates.get((lid,'mlp'))
            if gate is None and cpu_weights is None:return orig(x)
            idx=None if gate is None else (gate.reshape(-1)>=.5).nonzero().flatten()
            st['index']=idx;st['active']=True
            try:
                gw=weights(lid,'gate',m.gate_proj.weight,idx,0,x.device)
                uw=weights(lid,'up',m.up_proj.weight,idx,0,x.device)
                a=m.act_fn(F.linear(x,gw))*F.linear(x,uw)
                if idx is not None:
                    full=x.new_zeros(*x.shape[:-1],m.down_proj.weight.shape[1])
                    full.index_copy_(-1,idx,a)
                else:full=a
                return m.down_proj(full)
            finally:st['active']=False
        def down_forward(x,lid=layer_id,m=down,orig=original_down,st=state):
            if not st.get('active'):return orig(x)
            idx=st['index'];dw=weights(lid,'down',m.weight,idx,1,x.device)
            return F.linear(x if idx is None else x.index_select(-1,idx),dw,m.bias)
        originals.append((mlp,original_mlp,down,original_down))
        mlp.forward=forward;down.forward=down_forward
    try:yield
    finally:
        for mlp,forward,down,df in originals:mlp.forward=forward;down.forward=df


def copy_mlp_to_cpu(model,packed_down=False):
    rows=[]
    for l in model.model.layers:
        row={k:getattr(l.mlp,p).weight.detach().cpu().contiguous()
             for k,p in [('gate','gate_proj'),('up','up_proj'),('down','down_proj')]}
        if packed_down:row['down']=row['down'].t().contiguous();row['down_transposed']=True
        rows.append(row)
    return rows
