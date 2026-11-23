"""Export reproducible figures from the complete calibration/grid reports."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, root, name):
    for suffix in ['png','pdf']:
        fig.savefig(root/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('artifacts/sparsity-calibration-v2'))
    p.add_argument('--output',type=Path,default=Path('docs/sparsity-calibration'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cal=json.loads((a.input/'calibration-results.json').read_text())['results']
    kept=[.9,.75,.5]
    values=[cal[f'mlp-g1-magnitude-k{round(100*k)}-none']['overall'] for k in kept]
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].bar(range(3),[v['student_perplexity'] for v in values],color='#047857')
    axes[0].axhline(values[0]['dense_perplexity'],color='#334155',linestyle=':',label=f"Dense {values[0]['dense_perplexity']:.2f}")
    axes[0].set_title(f"Next-token perplexity (dense {values[0]['dense_perplexity']:.3f})")
    axes[1].bar(range(3),[v['teacher_kl_nats'] for v in values],color='#0e7490')
    axes[1].set_title('Teacher → student KL (nats/token)')
    for ax,key in zip(axes,['student_perplexity','teacher_kl_nats']):
        top=max(v[key] for v in values);ax.set_ylim(0,top*1.13)
        for i,v in enumerate(values):ax.text(i,v[key]+top*.02,f'{v[key]:.4g}',ha='center')
    for ax in axes:ax.set_xticks(range(3),['90%','75%','50%']);ax.set_xlabel('MLP neurons kept; attention stays dense')
    fig.suptitle('Calibration: keep the largest absolute neuron activations')
    save(fig,a.output,'neuron-calibration')
    gp=a.input/'grid-results.json'
    if not gp.exists():return
    grid=json.loads(gp.read_text())['results'];kept=[.9,.75,.5,.25,.1]
    rows=[('mlp',16),('mlp',32),('mlp',128),('attention',128),('both',16),('both',32),('both',128)]
    labels=['MLP · 16','MLP · 32','MLP · 128','Attention heads','Both · 16','Both · 32','Both · 128']
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    for ax,rule in zip(axes,['percentile','contribution','wina']):
        vals=np.array([[grid[f'{c}-g{g}-{rule}-k{round(100*k)}-none']['overall']['teacher_kl_nats']
                        for k in kept] for c,g in rows])
        im=ax.imshow(np.log10(np.maximum(vals,1e-4)),vmin=-4,vmax=2,cmap='magma',aspect='auto')
        for y in range(len(rows)):
            for x in range(len(kept)):
                ax.text(x,y,f'{vals[y,x]:.3g}',ha='center',va='center',fontsize=8,
                        color='white' if vals[y,x]<1 else 'black')
        ax.set_xticks(range(5),['90','75','50','25','10']);ax.set_yticks(range(7),labels)
        ax.set_xlabel('Percent kept within each component');ax.set_title(rule)
    fig.colorbar(im,ax=axes,label='log10 KL (nats/token)',shrink=.7)
    fig.suptitle('Component, selection rule and group size — no protections\nLive student activations; all layers affected; KL values printed in cells')
    save(fig,a.output,'component-grid')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors=['#64748b','#0e7490','#7c3aed','#047857']
    for ax,c in zip(axes,['mlp','attention','both']):
        g=128 if c=='attention' else 32
        for protection,color in zip(['none','always','edges','both'],colors):
            vals=[grid[f'{c}-g{g}-wina-k{round(100*k)}-{protection}']['overall']['teacher_kl_nats'] for k in kept]
            ax.plot([100*k for k in kept],np.maximum(vals,1e-6),marker='o',color=color,label=protection)
        ax.set_yscale('log');ax.set_title(c);ax.set_xlabel('Requested percent kept');ax.set_ylabel('KL (log scale)')
        ax.set_xticks([10,25,50,75,90]);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Protections: WINA-style selection, MLP groups of 32\nAlways-on counts inside budget; dense edge layers add computation')
    save(fig,a.output,'protections')


if __name__=='__main__':main()
