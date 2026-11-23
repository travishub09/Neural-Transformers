"""Export the fixed-run quality comparison; no model or checkpoint selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('docs/neuron-guessing')
    evaluation = json.loads((root / 'evaluation.json').read_text())
    methods = ['binary', 'hidden', 'fixed', 'oracle', 'dense_oracle',
               'previous_oracle', 'teal_mlp', 'wina_mlp']
    labels = ['Binary path predictor', 'Hidden-state predictor', 'Fixed selection',
              'Greedy magnitude', 'Dense-teacher masks*', 'Previous teacher mask*',
              'TEAL MLP reference', 'WINA MLP reference']
    colors = ['#187e72', '#366acc', '#75818a', '#c17a19', '#75818a', '#75818a', '#75818a', '#75818a']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    for ax, pct in zip(axes, [50, 40]):
        for i, method in enumerate(methods):
            row = evaluation['results'][f'{method}-{pct}']['overall']
            value = 100 * row['relative_ppl_increase']
            low, high = np.array(row['relative_ppl_increase_ci95']) * 100
            ax.plot([low, high], [i, i], color=colors[i], linewidth=2, alpha=.7)
            ax.scatter(value, i, color=colors[i], s=55, zorder=3)
            ax.annotate(f' {value:+.1f}%', (max(value, high), i),
                        xytext=(5, 0), textcoords='offset points', va='center', fontsize=9)
        ax.set_xscale('symlog', linthresh=1)
        ax.axvline(0, color='#a0a8ad', linewidth=1)
        ax.grid(axis='x', alpha=.2)
        ax.set_title(f'{pct}% of routed MLP neurons kept', loc='left', weight='bold', pad=15)
        ax.set_xlabel('Perplexity increase vs. dense (%)\nLinear near zero; logarithmic beyond ±1%')
        low, high = ax.get_xlim()
        ax.set_xlim(min(-.3, low), high * 2)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Neuron selection: quality compared with dense Qwen', x=.03,
                 ha='left', fontsize=17, weight='bold')
    fig.text(.03, .91, 'Frozen Qwen3-1.7B-Base · attention and first two MLP layers dense · 10M training tokens per predictor', fontsize=10)
    fig.text(.03, .035, 'Bars: 95% domain-stratified document-bootstrap intervals, one training seed. *Privileged teacher information.\n'
             'TEAL/WINA are restricted MLP references. The greedy magnitude selector is not a proven quality ceiling.', fontsize=9, color='#4b5760')
    fig.tight_layout(rect=(.02, .12, .98, .89))
    for extension in ['png', 'pdf']:
        fig.savefig(root / f'quality-comparison.{extension}', dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
