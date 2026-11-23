"""Batch-one teacher-forced latency; actual gathered projections and CPU copies."""
import argparse,sys,time
from contextlib import nullcontext
from pathlib import Path
import torch
from scripts.followup_common import backbone,planner_policy,source_rows,start_record,fingerprint,read,write,utc,ci
from scripts.mistral_comparison_core import Policy,score_batch
from pathpred.gathered_mlp import gathered_mlp,copy_mlp_to_cpu,TransferMeter
from scripts.followup_speculative import quality_summary


@torch.inference_mode()
def diagnose_gather(policy,tokens):
    """Validation-only fixed-input/fixed-mask parity, separate from timed work."""
    model=policy.model;ctrl=policy.ctrl;device=next(model.parameters()).device
    ids=torch.tensor([tokens[:64]],device=device);policy.reset();policy.prefill(ids,torch.ones_like(ids,dtype=torch.bool))
    captures={};hooks=[];layers=[2,15,31]
    for layer in layers:
        def capture(module,args,output,lid=layer):
            gate=ctrl.state.gates.get((lid,'mlp'))
            captures[lid]=(args[0].detach().clone(),output.detach().clone(),None if gate is None else gate.detach().clone())
        hooks.append(model.model.layers[layer].mlp.register_forward_hook(capture))
    try:policy.step(torch.tensor([[tokens[64]]],device=device),torch.ones(1,65,dtype=torch.bool,device=device))
    finally:
        for h in hooks:h.remove()
    result=[]
    for layer,(x,expected,mask) in captures.items():
        m=model.model.layers[layer].mlp;gates={} if mask is None else {(layer,'mlp'):mask}
        with gathered_mlp(model,ctrl),ctrl.use_gates(gates):actual=m(x)
        idx=torch.arange(m.gate_proj.weight.shape[0],device=device) if mask is None else (mask.reshape(-1)>=.5).nonzero().flatten()
        # FP32 checks math without converting or changing production weights.
        from torch.nn import functional as F
        gw=m.gate_proj.weight.float();uw=m.up_proj.weight.float();dw=m.down_proj.weight.float();xx=x.float()
        a=m.act_fn(F.linear(xx,gw))*F.linear(xx,uw)
        if mask is not None:a=a*mask
        dense=F.linear(a,dw);selected=F.linear(m.act_fn(F.linear(xx,gw[idx]))*F.linear(xx,uw[idx]),dw[:,idx])
        result.append(dict(layer=layer,kept_neurons=int(idx.numel()),fixed_mask_bf16_max_abs=float((actual-expected).abs().max()),fixed_mask_fp32_max_abs=float((selected-dense).abs().max()),fixed_mask_fp32_relative_rms=float((selected-dense).square().mean().sqrt()/dense.square().mean().sqrt().clamp_min(1e-12))))
    return dict(protocol='one reserved validation prefix, fixed inputs/masks, layers 2/15/31 zero-based; not used to select thresholds or kernels',layers=result)


@torch.inference_mode()
def timed(policy,prompt,continuation,implementation,cpu=None):
    device=next(policy.model.parameters()).device
    ids=torch.tensor([prompt],device=device);policy.reset()
    policy.prefill(ids,torch.ones_like(ids,dtype=torch.bool))
    meter=TransferMeter();kept=torch.zeros((),device=device);logits=[]
    context=gathered_mlp(policy.model,policy.ctrl,cpu_weights=cpu,meter=meter) if implementation=='gather' or cpu is not None else nullcontext()
    with context:
        torch.cuda.synchronize();start=time.perf_counter()
        for j,token in enumerate(continuation):
            z=policy.step(torch.tensor([[token]],device=device),torch.ones(1,len(prompt)+j+1,device=device,dtype=torch.bool))
            logits.append(z)
            masks=policy.masks();kept+=1 if masks is None else masks.float().mean()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
    return dict(seconds=elapsed,tokens=len(continuation),inference_tokens_per_second=len(continuation)/elapsed,
        routed_kept=float(kept)/len(continuation),cpu_to_device_weight_payload_bytes=meter.bytes,
        cpu_to_device_weight_payload_bytes_per_token=meter.bytes/len(continuation)),torch.stack(logits)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/followup-latency.json');a=p.parse_args();c=read(a.config)
    root=Path(c['collection']);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False
    identity=start_record(root,sys.argv);fingerprint(root)
    try:
        sources,manifest=source_rows();write(root/'data-manifest.json',manifest)
        model,ctrl=backbone();dense=Policy(model,ctrl,'dense');cpu=copy_mlp_to_cpu(model,packed_down=c.get('packed_cpu_down',False))
        conditions={};test=sources['test'][:c['documents']];warm=sources['validation'][0]['token_ids']
        # Fixed teacher-forced inputs ensure every timing comparison does the same work.
        dense_reference={};dense_logits={}
        for storage in ['gpu','cpu']:
            for budget in [100]+c['budgets']:
                if budget==100:pol=dense
                else:pol,prov=planner_policy(model,ctrl,budget);write(root/f'checkpoint-{budget}.json',prov)
                if storage=='gpu' and budget!=100 and c.get('validation_parity_diagnostic'):
                    write(out/f'validation-parity-{budget}.json',diagnose_gather(pol,warm))
                implementations=['dense'] if budget==100 else ['masked','gather'] if storage=='gpu' else ['gather']
                for impl in implementations:
                    name=f'{storage}-{budget}-{impl}';results=[];quality=[]
                    timed(pol,warm[:64],warm[64:68],impl,cpu if storage=='cpu' else None)
                    for row in test:
                        prompt=row['token_ids'][:c['prompt_tokens']];suffix=row['token_ids'][c['prompt_tokens']:c['prompt_tokens']+c['tokens']]
                        # Independent untimed dense-masked reference for numerical parity.
                        _,reference=timed(pol,prompt,suffix,'dense' if budget==100 else 'masked')
                        for repeat in range(c['repetitions']):
                            value,z=timed(pol,prompt,suffix,impl,cpu if storage=='cpu' else None)
                            value.update(document=row['document_sha256'],repeat=repeat,
                                max_absolute_logit_difference=float((z-reference).abs().max()),
                                top1_agreement_vs_same_policy_masked=float((z.argmax(-1)==reference.argmax(-1)).float().mean()))
                            results.append(value);write(out/f'{name}-documents.json',results)
                        key=row['document_sha256']
                        if budget==100:dense_logits[key]=z.detach()
                        tp=dense_logits[key].float().log_softmax(-1);sp=z.float().log_softmax(-1)
                        targets=torch.tensor(row['token_ids'][c['prompt_tokens']+1:c['prompt_tokens']+c['tokens']+1],device=z.device)[:,None,None]
                        quality.append(dict(document=key,domain='wikitext',tokens=c['tokens'],
                            teacher_nll=float(-tp.gather(-1,targets).sum()),student_nll=float(-sp.gather(-1,targets).sum()),
                            kl=float((tp.exp()*(tp-sp)).sum()),top1_agreement=float((tp.argmax(-1)==sp.argmax(-1)).sum()),
                            kept_sum=value['routed_kept']*c['tokens'],executions=c['tokens']))
                        write(out/f'{name}-quality-documents.json',quality)
                        print(name,len(results),flush=True)
                    by_doc=[dict(document=r['document_sha256'],seconds=sum(x['seconds'] for x in results if x['document']==r['document_sha256'])/c['repetitions']) for r in test]
                    conditions[name]=dict(inference_tokens_per_second=sum(x['tokens'] for x in results)/sum(x['seconds'] for x in results),
                        routed_kept=sum(x['routed_kept'] for x in results)/len(results),
                        cpu_to_device_weight_payload_bytes_per_token=sum(x['cpu_to_device_weight_payload_bytes'] for x in results)/sum(x['tokens'] for x in results),
                        max_absolute_logit_difference=max(x['max_absolute_logit_difference'] for x in results),
                        min_top1_parity=min(x['top1_agreement_vs_same_policy_masked'] for x in results),by_document=by_doc)
                    qs=quality_summary(quality)
                    for metric in ['student_perplexity','dense_perplexity','relative_ppl_increase','teacher_kl_nats','top1_agreement','routed_kept']:
                        qs[metric+'_ci95']=ci(quality,lambda rs:quality_summary(rs)[metric])
                    conditions[name]['quality']=qs
                    conditions[name]['strict_top1_parity_passed']=conditions[name]['min_top1_parity']==1.0
                    if budget==100:dense_reference[storage]=by_doc
                    else:
                        pairs=list(zip(dense_reference[storage],by_doc))
                        ratio=lambda ps:sum(x[0]['seconds'] for x in ps)/sum(x[1]['seconds'] for x in ps)
                        conditions[name].update(speedup_vs_dense=ratio(pairs),speedup_ci95=ci(pairs,ratio))
                        conditions[name]['speedup_supported']=conditions[name]['speedup_ci95'][0]>1 and conditions[name]['strict_top1_parity_passed']
                    write(out/'summary.json',conditions)
                if budget!=100:del pol;torch.cuda.empty_cache()
        write(root/'completion.json',dict(status='complete',finished_utc=utc(),reason='All timing conditions measured; review pending.'))
    except BaseException as e:
        write(root/'completion.json',dict(status='failed',finished_utc=utc(),reason=repr(e)));raise
    finally:
        write(root/'compute.json',dict(peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            throughput_kind='inference',cpu_weight_payload_caveat='Actual copied tensor bytes; NOT measured CPU DRAM transactions. Full GPU weight shadows remain for dense prompts/reference; no reduced-VRAM capacity claim.'))
        identity['end_utc']=utc();write(root/'identity.json',identity);fingerprint(root)


if __name__=='__main__':main()
