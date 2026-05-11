"""Run cross-dataset experiments."""









import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
from functools import partial

from project.config import FIGURES_DIR
from project.experiments import main_experiment as M

# Experiment settings
N_AGENTS       = 40
N_TRIALS       = 50
TASK_NUM_RANGE = [20, 40, 60, 80, 100, 120]
ALGS           = ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']
OUTPUT_DIR     = str(FIGURES_DIR)


COLORS = {
    'CHASE': '#E53935',
    'MCT':   '#FB8C00',
    'IRS':   '#26A69A',
    'DyLAN': '#78909C',
    'FL_DRL':'#7E57C2',
    'BRTOA': '#00897B',
}
MARKERS = {'CHASE': 'o', 'MCT': 's', 'IRS': '^', 'DyLAN': 'x', 'FL_DRL': 'd', 'BRTOA': 'p'}
STYLES  = {
    'CHASE':  '-',
    'MCT':    '--',
    'IRS':    '-.',
    'DyLAN':  ':',
    'FL_DRL': (0, (3, 1, 1, 1)),
    'BRTOA':  (0, (5, 2)),
}
LABELS = {
    'CHASE': 'CHASE (Proposed)', 'MCT': 'MCT',
    'IRS': 'IRS (Random)',       'DyLAN': 'DyLAN',
    'FL_DRL': 'FL-DRL',          'BRTOA': 'BRTOA',
}

# Worker returns both metrics
def _worker(trial_idx, alg, n_task, alibaba_data, real_agent_dist):
    random.seed()
    np.random.seed()
    agents = M.create_agents_from_real_data(N_AGENTS, real_distribution=real_agent_dist)
    tasks  = M.create_tasks_from_real_data(n_task, real_data=alibaba_data)
    res    = M.run_algorithm(alg, agents, tasks)
    return res['revenue'], res['success_rate']


def run_experiment(pool, alibaba_data, real_agent_dist):








    results = {
        alg: {'rev_means': [], 'rev_stds': [], 'succ_means': [], 'succ_stds': []}
        for alg in ALGS
    }

    for n_task in TASK_NUM_RANGE:
        print(f"  Tasks={n_task} ...", flush=True)
        for alg in ALGS:
            fn = partial(_worker,
                         alg=alg, n_task=n_task,
                         alibaba_data=alibaba_data,
                         real_agent_dist=real_agent_dist)
            trial_out = pool.map(fn, range(N_TRIALS))

            revs  = np.array([r for r, _ in trial_out])
            succs = np.array([s for _, s in trial_out])

            results[alg]['rev_means'].append(float(revs.mean()))
            results[alg]['rev_stds'].append(float(revs.std()))
            results[alg]['succ_means'].append(float(succs.mean()))
            results[alg]['succ_stds'].append(float(succs.std()))

            print(f"    {alg:8s}  rev={revs.mean():.1f}±{revs.std():.1f}"
                  f"  succ={succs.mean()*100:.1f}%±{succs.std()*100:.1f}%")

    return results


# Shared plotting helper
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

DRAW_ORDER = ['MCT', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA', 'CHASE']


def _plot_line(x_data, y_means_dict, y_stds_dict, ylabel, title, filename):

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for alg in DRAW_ORDER:
        y_mean = np.array(y_means_dict[alg])
        y_std  = np.array(y_stds_dict[alg])

        lw     = 3.5 if alg == 'CHASE' else 2.0
        ms     = 10  if alg == 'CHASE' else 7
        zorder = 10  if alg == 'CHASE' else 5
        ls     = STYLES[alg]

        ax.errorbar(x_data, y_mean,
                    yerr=y_std,
                    marker=MARKERS[alg],
                    linestyle=ls,
                    color=COLORS[alg],
                    label=LABELS[alg],
                    linewidth=lw,
                    markersize=ms,
                    markeredgecolor='white',
                    markeredgewidth=0.8,
                    zorder=zorder,
                    capsize=4,
                    elinewidth=1.2,
                    capthick=1.2,
                    ecolor=COLORS[alg],
                    alpha=0.85)

    ax.set_xlabel('Number of Tasks', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=12, pad=10)

    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc',
              fontsize=10, loc='best')

    ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color='#999999')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    ax.set_xticks(x_data)
    ax.set_xticklabels([str(int(t)) for t in x_data])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f">>> Saved: {filename}")
    return path


def plot_all(results):
    x_data = np.array(TASK_NUM_RANGE)


    _plot_line(
        x_data,
        y_means_dict={alg: results[alg]['rev_means'] for alg in ALGS},
        y_stds_dict ={alg: results[alg]['rev_stds']  for alg in ALGS},
        ylabel   = 'Total Revenue',
        title    = 'Total Revenue vs. Number of Tasks\n(Alibaba Cluster Trace, Agents=40)',
        filename = 'Fig_CrossDataset_Alibaba_Revenue.png',
    )



    _plot_line(
        x_data,
        y_means_dict={alg: [v * 100 for v in results[alg]['succ_means']] for alg in ALGS},
        y_stds_dict ={alg: [v * 100 for v in results[alg]['succ_stds']]  for alg in ALGS},
        ylabel   = 'Success Rate (%)',
        title    = 'Task Success Rate vs. Number of Tasks\n(Alibaba Cluster Trace, Agents=40)',
        filename = 'Fig_CrossDataset_Alibaba_SuccRate.png',
    )


# Main workflow
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(">>> Loading Alibaba synthetic data...")
    alibaba_data = M.load_alibaba_synthetic_data()

    print(">>> Loading agent distribution from Borg trace...")
    agent_dist = M.load_agent_distribution()

    num_workers = min(max(1, multiprocessing.cpu_count() - 2), 8)
    print(f">>> Using {num_workers} worker processes.")
    print(f">>> Task range: {TASK_NUM_RANGE}, Trials: {N_TRIALS}\n")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = run_experiment(pool, alibaba_data, agent_dist)

    print()
    plot_all(results)
    print(">>> Cross-dataset (Alibaba) experiment complete.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
