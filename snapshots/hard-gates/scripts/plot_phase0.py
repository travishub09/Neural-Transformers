"""Standalone publication-style descriptive plots from frozen Phase 0 results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = json.loads(a.results.read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180})
    names = {'decisions': 'Decisions only', 'binary': 'Binary path', 'coarse': 'Three levels',
             'magnitudes': 'Exact magnitudes', 'hidden': 'Hidden state',
             'frequency': 'Static frequency', 'mean_norm': 'Static norm', 'oracle': 'Oracle (labels known)'}
    colors = {'decisions': '#94a3b8', 'binary': '#047857', 'coarse': '#0e7490',
              'magnitudes': '#7c3aed', 'hidden': '#d97706', 'frequency': '#334155',
              'mean_norm': '#334155', 'oracle': '#cbd5e1'}
    primary = r['results'][str(r['decision']['primary_budget'])]
    methods = ['frequency', 'mean_norm', 'decisions', 'binary', 'coarse', 'magnitudes', 'hidden', 'oracle']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), layout='constrained')
    for ax, metric, title in zip(axes, ['recall', 'norm_mass_fraction'],
                                ['Recall of norm-coverage targets', 'Family-balanced norm mass retained']):
        values = [100*primary['methods'][s]['mean'][metric] for s in methods]
        ax.barh(np.arange(len(methods)), values, color=[colors[s] for s in methods])
        ax.set_yticks(np.arange(len(methods)), [names[s] for s in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, max(values)*1.18)
        ax.set_xlabel('Percent')
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(value+.25, i, f'{value:.2f}', va='center', fontsize=9)
    fig.suptitle('Phase 0: 25% gated-budget anchor, matched total modeled cost\n'
                 'Dense-activation proxies; neither language-model quality nor inference speed', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(a.output / ('phase0-primary.'+suffix), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout='constrained')
    budgets = [.1, .25, .5]
    for signal in ['binary', 'coarse', 'magnitudes', 'hidden', 'static', 'oracle']:
        means, errors = [], []
        for b in budgets:
            entry = r['results'][str(b)]
            name = entry['selected_static'] if signal == 'static' else signal
            m = entry['methods'][name]
            means.append(100*m['mean']['recall'])
            errors.append(100*m.get('recall_seed_sd', 0))
        color = '#334155' if signal == 'static' else colors[signal]
        label = 'Validation-selected static' if signal == 'static' else names[signal]
        ax.errorbar(np.array(budgets)*100, means, yerr=errors, marker='o', label=label,
                    color=color, linestyle='--' if signal in ['static', 'oracle'] else '-')
    ax.set_xlabel('Gated-budget anchor (%) — not total model compute')
    ax.set_ylabel('Norm-target recall (%)')
    ax.set_title('Held-out proxy prediction across three budgets\nMean ± training-seed SD; common total cost cap within each budget')
    ax.set_xticks([10, 25, 50])
    ax.legend(frameon=False, fontsize=9)
    for suffix in ['png', 'pdf']:
        fig.savefig(a.output / ('phase0-budgets.'+suffix), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
