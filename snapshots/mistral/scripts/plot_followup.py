"""Standalone figures for activity-oracle quality, granularity and prediction."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def save(fig, output, name):
    for suffix in ['png','pdf']:
        fig.savefig(output/(name+'.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('artifacts'))
    p.add_argument('--output',type=Path,default=Path('docs/step6-followup'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    quality=json.loads((a.root/'oracle-quality-v1/results.json').read_text())
    q=quality['results'];budgets=[10,25,50]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    palette={'oracle_binary':'#047857','oracle_ranked':'#7c3aed','fixed_mean_norm':'#334155','fixed_frequency':'#64748b','random':'#d97706'}
    names={'oracle_binary':'Binary activity oracle','oracle_ranked':'Strength-ranked activity oracle',
           'fixed_mean_norm':'Fixed mean norm','fixed_frequency':'Fixed strong frequency','random':'Random'}
    for name,color in palette.items():
        for ax,metric in zip(axes,['student_perplexity','teacher_kl_nats']):
            values=[q[name+'-b'+str(b)]['overall'][metric] for b in budgets]
            ax.plot(budgets,values,marker='o',color=color,label=names[name])
    dense=q['dense']['overall']['dense_perplexity']
    axes[0].axhline(dense,color='#111827',linestyle=':',label=f'Dense: {dense:.2f}')
    axes[1].axhline(quality['config']['quality_max_teacher_kl_nats'],color='#111827',linestyle=':',label='KL quality limit')
    axes[1].text(10,quality['config']['quality_max_teacher_kl_nats']*1.12,'Provisional KL limit: 0.05',fontsize=9)
    for ax,metric,title in zip(axes,['student_perplexity','teacher_kl_nats'],['Next-token perplexity (log scale)','Teacher → student KL, nats/token (log scale)']):
        x=100*q['literal_q75']['budget_observed']['selected_units_fraction']
        y=q['literal_q75']['overall'][metric]
        ax.scatter([x],[y],marker='D',s=50,color='#dc2626',label=f'Literal q75 mask ({x:.1f}% realized)')
        ax.set_yscale('log');ax.set_xticks(budgets);ax.set_xlabel('Global gated-unit budget (%)')
        ax.set_title(title);ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('Activity oracles do not preserve the frozen model’s output\n63 documents; first two layers and KV remain dense; no gate rescaling',fontsize=12)
    save(fig,a.output,'oracle-quality')
    gran=json.loads((a.root/'granularity-v1/results.json').read_text())['results']['by_split']['test']['metrics']
    groups=['neurons','learned_subgroups32','learned128','original128','random128']
    labels=['Individual neurons','Learned groups subdivided to 32','Learned groups of 128','Original groups of 128','Random groups of 128']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,metric,title in zip(axes,['energy_in_top_25pct','fraction_for_95pct_energy'],
        ['Energy captured by the largest 25%','Units needed to capture 95% of energy']):
        values=[100*gran[g+'__'+metric]['mean'] for g in groups]
        ax.barh(range(len(groups)),values,color=['#047857','#0e7490','#7c3aed','#94a3b8','#64748b'])
        ax.set_yticks(range(len(groups)),labels);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xlabel('Percent');ax.set_title(title)
        for i,v in enumerate(values):ax.text(v+.8,i,f'{v:.1f}',va='center')
    fig.suptitle('Grouping hides substantial neuron-level energy concentration\nHeld-out sampled positions; squared energy is additive; concentration is not output quality',fontsize=12)
    save(fig,a.output,'granularity')
    result_path=a.root/'phase0-strong-v1/results.json'
    if not result_path.exists():return
    r=json.loads(result_path.read_text())['results']
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    names={'decisions':'Decisions only','binary':'Binary path','coarse':'Three levels','magnitudes':'Exact magnitudes','hidden':'Hidden state'}
    colors={'decisions':'#94a3b8','binary':'#047857','coarse':'#0e7490','magnitudes':'#7c3aed','hidden':'#d97706'}
    for name in names:
        metrics=[r[str(b/100)]['normalizations'][name]['normalized_skill'] for b in budgets]
        values=np.array([m['value'] for m in metrics]);ci=np.array([m['ci95'] for m in metrics])
        axes[0].errorbar(budgets,values,yerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),marker='o',label=names[name],color=colors[name])
    axes[0].axhline(0,color='#334155',linestyle=':',label='Expected random = 0')
    axes[0].axhline(1,color='#334155',linestyle='--',label='Perfect label ranking = 1')
    axes[0].set_title('Strong-bit prediction skill\nNormalized at each method’s selection counts')
    axes[0].set_ylabel('(Method − random) / (oracle − random)')
    metrics=[r[str(b/100)]['normalizations']['binary']['fraction_of_oracle_fixed_advantage_at_common_cap'] for b in budgets]
    values=np.array([m['value'] for m in metrics]);ci=np.array([m['ci95'] for m in metrics])
    axes[1].errorbar(budgets,values,yerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),marker='o',color=colors['binary'])
    axes[1].axhline(.2,color='#334155',linestyle=':',label='Provisional 20% prediction gate')
    axes[1].axhline(0,color='#94a3b8',linewidth=.7)
    axes[1].set_title('Binary path: net gain over fixed selection\nCommon total modeled cost cap')
    axes[1].set_ylabel('(Binary − fixed) / (oracle at static cap − fixed)')
    for ax in axes:
        ax.set_xticks(budgets);ax.set_xlabel('Gated-budget anchor (%)');ax.legend(frameon=False,fontsize=8)
    axes[0].legend(frameon=False,fontsize=8,loc='lower right',bbox_to_anchor=(1,.12))
    fig.suptitle('Revised Phase 0: activity predictability is separate from safe pruning\nPaired seed/document bootstrap 95% intervals; quality gate must also pass',fontsize=12)
    save(fig,a.output,'strong-prediction')


if __name__=='__main__':main()
