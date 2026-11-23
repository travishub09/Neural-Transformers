"""Quality curves for available validation-calibrated checkpoints."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=False)
    for ax,budget,reference in zip(axes,[50,40],[.9568,2.390]):
        ax.axhline(reference,color='#777777',linestyle='--',lw=1.3,label='Magnitude reference')
        for name,root,initial,color in [
            ('Binary path','e2e-full-v1','native-calibration-v1','#167d8d'),
            ('Hidden state','e2e-hidden-full-v1','hidden-native-calibration-v1','#a54dce')]:
            points=[]
            first=Path(f'artifacts/{initial}/evaluation-{budget}.json')
            if first.exists():points.append((0,json.loads(first.read_text())['overall']))
            for path in sorted(Path(f'artifacts/{root}/budget-{budget}').glob('evaluation-*.json')):
                if path.stem.endswith('-documents'):continue
                points.append((int(path.stem.removeprefix('evaluation-'))/1e6,json.loads(path.read_text())['overall']))
            if not points:continue
            x=[t for t,o in points];y=[100*o['relative_ppl_increase'] for t,o in points]
            lower=[v-100*o['relative_ppl_increase_ci95'][0] for v,(_,o) in zip(y,points)]
            upper=[100*o['relative_ppl_increase_ci95'][1]-v for v,(_,o) in zip(y,points)]
            ax.errorbar(x,y,yerr=[lower,upper],marker='o',lw=2,color=color,capsize=3,label=name)
        ax.set_title(f'{budget}% validation retention target',fontweight='bold')
        ax.set_xlabel('Additional training tokens (millions)');ax.set_xlim(-.4,10.4)
        ax.set_ylabel('Perplexity increase vs dense (%)');ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    fig.text(.5,.02,'Hard gates; test retention may differ from target. Error bars: document bootstrap, not training-seed variability.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1]);out=Path('docs/e2e-matched');out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'quality-curves.{ext}',dpi=200,bbox_inches='tight')


if __name__=='__main__':main()
