"""Reconstruct paired quality results and export the quick comparison figure."""
import json,math,hashlib,struct
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('artifacts/fresh-groups-quick-v1')
r=json.loads((root/'results.json').read_text())
m=json.loads((root/'manifest.json').read_text())
docs=[json.loads(p.read_text()) for p in root.glob('window-*.json')]
assert len(docs)==63
tensor_count=0
for layer in m['layers']:
    for item in layer['sizes'].values():
        f=item['file'];blob=(root/f['file']).read_bytes()
        assert len(blob)==f['bytes'] and hashlib.sha256(blob).hexdigest()==f['sha256']
        header_length=struct.unpack('<Q',blob[:8])[0]
        header=json.loads(blob[8:8+header_length]);data=blob[8+header_length:]
        for key in ['permutation','stored_indices']:
            meta=header[key];assert meta['dtype']=='I64'
            values=np.frombuffer(data[slice(*meta['data_offsets'])],dtype='<i8')
            assert np.array_equal(np.sort(values),np.arange(6144))
        tensor_count+=1
assert tensor_count==56 and all(r['checks'].values())
for key,v in r['results'].items():
    rows=[d['conditions'][key] for d in docs];n=sum(x['tokens'] for x in rows)
    for metric,col in [('teacher_kl_nats','kl'),('dense_nll','teacher_nll'),('student_nll','student_nll')]:
        assert math.isclose(v['overall'][metric],sum(x[col] for x in rows)/n,rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(v['overall']['student_perplexity'],math.exp(v['overall']['student_nll']),rel_tol=1e-12)
for key,v in r['paired_comparisons'].items():
    old=r['results'][key+'-old_subdivision']['overall'];new=r['results'][key+'-fresh']['overall']
    assert math.isclose(v['fresh_vs_old_relative_ppl_change'],new['student_perplexity']/old['student_perplexity']-1,abs_tol=1e-12)
    assert math.isclose(v['fresh_minus_old_kl'],new['teacher_kl_nats']-old['teacher_kl_nats'],abs_tol=1e-12)
summary={}
for size in ['16','32']:
    items=[l['sizes'][size] for l in m['layers']]
    summary[size]={'layers':len(items),'hit_iteration_cap':sum(x['fit']['runs'][0]['stop_reason']=='iteration_limit' for x in items),
        'mean_iterations':sum(len(x['fit']['runs'][0]['objectives']) for x in items)/len(items),
        'mean_fraction_crossing_old_boundaries':sum(x['fraction_new_groups_crossing_old_128_boundaries'] for x in items)/len(items),
        'heldout_cosine':{name:sum(x['metrics']['test'][name]['mean_pairwise_magnitude_cosine'] for x in items)/len(items) for name in ['old_subdivision','fresh']}}
out=Path('docs/fresh-groups-quick');out.mkdir(parents=True,exist_ok=True)
(out/'fit-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'independent-audit.json').write_text(json.dumps({'passed':True,'documents':63,'conditions':8,'paired_comparisons':4,'metrics_independently_reconstructed':True,'tensor_hashes_and_bijections_verified':tensor_count},indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
keys=['g16-k75','g16-k50','g32-k75','g32-k50'];labels=['16 / 75%','16 / 50%','32 / 75%','32 / 50%']
for ax,metric,title in zip(axes,['student_perplexity','teacher_kl_nats'],['Perplexity','KL from dense model (nats/token)']):
    for offset,name,label,color in [(-.18,'old_subdivision','Old subdivisions','#94a3b8'),(.18,'fresh','Fresh clusters','#047857')]:
        vals=[r['results'][k+'-'+name]['overall'][metric] for k in keys]
        ax.bar([i+offset for i in range(4)],vals,width=.36,label=label,color=color)
    ax.set_xticks(range(4),labels);ax.set_xlabel('Group size / fraction kept');ax.set_title(title);ax.spines[['top','right']].set_visible(False)
axes[0].axhline(r['results'][keys[0]+'-fresh']['overall']['dense_perplexity'],color='#334155',linestyle=':',label='Dense')
axes[0].legend(frameon=False,fontsize=8)
fig.suptitle('Fresh balanced groups versus subdivisions of old 128-neuron groups\nMLP only; projected contribution; attention dense; identical 63 documents')
for suffix in ['png','pdf']:fig.savefig(out/('comparison.'+suffix),dpi=180,bbox_inches='tight')
plt.close(fig)
print(json.dumps({'audit':'passed','fit_summary':summary}))
