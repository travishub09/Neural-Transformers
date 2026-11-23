"""Export fixed-budget quality and native-policy retention across checkpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('docs/e2e-pilot')
    runs=[json.loads(p.read_text()) for p in sorted(root.glob('evaluation-*.json'))]
    x=[r['trained_tokens']/1e6 for r in runs]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,5))
    colors={'topk-50':'#187e72','topk-40':'#366acc'}
    for name in ['topk-50','topk-40']:
        values=[r['conditions'][name]['overall'] for r in runs]
        axes[0].plot(x,[v['relative_ppl_increase']*100 for v in values],'-o',color=colors[name],label=name.replace('topk-','')+'% kept')
        axes[0].fill_between(x,[v['relative_ppl_increase_ci95'][0]*100 for v in values],
                            [v['relative_ppl_increase_ci95'][1]*100 for v in values],color=colors[name],alpha=.12)
    axes[0].set_title('Exact hard top-k budgets',loc='left',weight='bold')
    axes[0].set_ylabel('Perplexity increase vs. dense (%)');axes[0].legend(frameon=False)
    native=[r['conditions']['native']['overall'] for r in runs]
    axes[1].plot(x,[v['relative_ppl_increase']*100 for v in native],'-o',color='#9a6416')
    axes[1].fill_between(x,[v['relative_ppl_increase_ci95'][0]*100 for v in native],
                        [v['relative_ppl_increase_ci95'][1]*100 for v in native],color='#9a6416',alpha=.12)
    axes[1].set_title('Native hard-threshold quality',loc='left',weight='bold')
    axes[1].set_ylabel('Perplexity increase vs. dense (%)')
    axes[2].plot(x,[v['routed_kept']*100 for v in native],'-o',color='#9a6416',label='Actual kept')
    axes[2].plot([0,2],[50,50],'--',color='#7d8990',label='Training target')
    axes[2].plot([2,3],[40,40],'--',color='#7d8990')
    axes[2].set_title('Native hard-threshold budget',loc='left',weight='bold')
    axes[2].set_ylabel('Routed MLP neurons kept (%)');axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel('End-to-end training tokens (millions)');ax.set_xticks([0,1,2,3]);ax.grid(alpha=.2)
        ax.axvline(2,color='#aab2b7',linewidth=1,linestyle=':')
    fig.suptitle('End-to-end quality training from the binary predictor',x=.025,ha='left',fontsize=17,weight='bold')
    fig.text(.025,.89,'Frozen Qwen3-1.7B-Base · dense attention and first two layers · 2M tokens at 50%, then 1M at 40%',fontsize=10)
    fig.text(.025,.025,'All evaluations use hard gates on the same 63 documents. Shading: 95% document-bootstrap intervals; one training seed.\n'
             'Native quality must be read alongside actual retention. No sparse-kernel speedup is measured.',fontsize=9,color='#4b5760')
    fig.tight_layout(rect=(.015,.1,.99,.88))
    for ext in ['png','pdf']:fig.savefig(root/f'checkpoint-comparison.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
