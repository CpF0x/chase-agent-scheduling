"""Run revenue-parameter sensitivity experiments."""





















import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import multiprocessing
from functools import partial

from after_project.config import FIGURES_DIR
from after_project.experiments import main_experiment as M

# Experiment settings
N_TASK   = 80
N_AGENTS = 40
N_TRIALS = 30


DELTA_SWEEP = [5, 10, 20, 30, 50, 80]
GAMMA_SWEEP = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


DELTA_BASE = 50.0
GAMMA_BASE = 3.0

OUTPUT_DIR = str(FIGURES_DIR)


_BASE_TAU_K = M.calc_tau(100.0, M.REVENUE_COST, DELTA_BASE, GAMMA_BASE)


MAX_REVENUE_PER_TASK = 100.0 - M.REVENUE_COST
MAX_REVENUE_TOTAL    = N_TASK * MAX_REVENUE_PER_TASK

print(f"[INFO] Baseline τ_k = {_BASE_TAU_K:.6f}")
print(f"[INFO] effective_deadline (fixed) = {M.DEADLINE - _BASE_TAU_K:.6f}")
print(f"[INFO] max possible revenue = {MAX_REVENUE_TOTAL:.1f}")


# Per-trial worker
def _worker(trial_idx, delta, gamma, real_trace_data, real_agent_dist):









    random.seed()
    np.random.seed()


    M.REVENUE_DELTA = delta
    M.REVENUE_GAMMA = gamma


    agents = M.create_agents_from_real_data(N_AGENTS, real_distribution=real_agent_dist)
    tasks  = M.create_tasks_from_real_data(N_TASK, real_data=real_trace_data)


    for t in tasks:
        t.tau_k = _BASE_TAU_K


    result = M.run_algorithm('CHASE', agents, tasks)

    raw_revenue = result['revenue']
    efficiency  = raw_revenue / MAX_REVENUE_TOTAL
    return raw_revenue, efficiency


def run_grid_point(pool, delta, gamma, real_trace_data, real_agent_dist):

    fn = partial(_worker,
                 delta=delta, gamma=gamma,
                 real_trace_data=real_trace_data,
                 real_agent_dist=real_agent_dist)
    results = pool.map(fn, range(N_TRIALS))
    revs = np.array([r[0] for r in results])
    return revs.mean(), revs.std()


# Plot settings
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.serif'       : ['Times New Roman'],
    'font.size'        : 11,
    'axes.labelsize'   : 13,
    'axes.titlesize'   : 13,
    'legend.fontsize'  : 10,
    'xtick.labelsize'  : 11,
    'ytick.labelsize'  : 11,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
})

PRIMARY  = '#0868a6'
CHOSEN   = '#E65100'
BAND_A   = 0.18



def plot_1d(x_vals, means, stds, xlabel, title, chosen_x, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_vals, means, marker='o', color=PRIMARY,
            linewidth=2.5, markersize=8,
            markeredgecolor='white', markeredgewidth=0.8, zorder=5)
    ax.fill_between(x_vals,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=PRIMARY, alpha=BAND_A, zorder=4)


    idx = x_vals.index(chosen_x)
    ax.axvline(chosen_x, linestyle='--', linewidth=1.2, color=CHOSEN, zorder=3)
    ax.annotate(f'Chosen: {chosen_x}',
                xy=(chosen_x, means[idx]),
                xytext=(15, 8), textcoords='offset points',
                fontsize=10, color=CHOSEN,
                arrowprops=dict(arrowstyle='->', color=CHOSEN, lw=1.1),
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFF8E1',
                          ec='#FB8C00', lw=0.8))

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('Mean Total Revenue (CHASE)', fontsize=13)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(x_vals)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  >>> Saved: {fname}")


# Heatmap plot
def plot_heatmap(delta_vals, gamma_vals, rev_grid, fname):




    fig, ax = plt.subplots(figsize=(8, 5))


    data = rev_grid.T

    im = ax.imshow(data, aspect='auto', origin='lower',
                   cmap='YlOrRd',
                   extent=[0, len(delta_vals), 0, len(gamma_vals)])
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Mean Total Revenue', fontsize=12)


    ax.set_xticks(np.arange(len(delta_vals)) + 0.5)
    ax.set_yticks(np.arange(len(gamma_vals)) + 0.5)
    ax.set_xticklabels([str(d) for d in delta_vals], fontsize=10)
    ax.set_yticklabels([f'{g:.1f}' for g in gamma_vals], fontsize=10)
    ax.set_xlabel(r'$\delta$ (sigmoid steepness)', fontsize=12)
    ax.set_ylabel(r'$\gamma$ (convexity exponent)', fontsize=12)


    vmax = data.max()
    for i in range(len(delta_vals)):
        for j in range(len(gamma_vals)):
            val = data[j, i]
            ax.text(i + 0.5, j + 0.5, f'{val:.0f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if val > vmax * 0.75 else 'black')


    bi, bj = np.unravel_index(np.argmax(rev_grid), rev_grid.shape)
    ax.plot(bi + 0.5, bj + 0.5, marker='*', markersize=22,
            markerfacecolor='none', markeredgecolor='white', markeredgewidth=1.8,
            label=f'Best: δ={delta_vals[bi]}, γ={gamma_vals[bj]}', zorder=5)


    ci = delta_vals.index(DELTA_BASE)
    cj = gamma_vals.index(GAMMA_BASE)
    ax.plot(ci + 0.5, cj + 0.5, marker='D', markersize=14,
            markerfacecolor='none', markeredgecolor='#90CAF9', markeredgewidth=2.0,
            label=f'Chosen: δ={int(DELTA_BASE)}, γ={GAMMA_BASE}', zorder=5)

    ax.legend(fontsize=10, loc='upper right',
              frameon=True, facecolor='white', framealpha=0.85)

    ax.set_title(
        f'Sensitivity Heatmap: CHASE Mean Revenue vs ($\\delta$, $\\gamma$)\n'
        f'(Fixed $\\tau_k$={_BASE_TAU_K:.4f}, Tasks={N_TASK}, Agents={N_AGENTS}, '
        f'{N_TRIALS} trials)',
        fontsize=12, pad=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  >>> Saved: {fname}")


# Main workflow
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(">>> Loading real trace data...")
    real_trace  = M.load_real_trace_data()
    real_agents = M.load_agent_distribution()

    num_workers = min(max(1, multiprocessing.cpu_count() - 2), 8)
    print(f">>> Using {num_workers} worker processes, {N_TRIALS} trials per point.")


    rev_grid = np.zeros((len(DELTA_SWEEP), len(GAMMA_SWEEP)))
    std_grid = np.zeros((len(DELTA_SWEEP), len(GAMMA_SWEEP)))


    means_d, stds_d = [], []
    means_g, stds_g = [], []

    with multiprocessing.Pool(processes=num_workers) as pool:


        print("\n>>> 2D Grid sweep...")
        total = len(DELTA_SWEEP) * len(GAMMA_SWEEP)
        done  = 0
        for i, d in enumerate(DELTA_SWEEP):
            for j, g in enumerate(GAMMA_SWEEP):
                mr, sr = run_grid_point(pool, d, g, real_trace, real_agents)
                rev_grid[i, j] = mr
                std_grid[i, j] = sr
                done += 1
                print(f"  [{done:>2d}/{total}] δ={d:5.0f}, γ={g:.1f}  rev={mr:.1f}±{sr:.1f}")


        print("\n>>> Extracting 1D sweeps from grid...")
        gi_g3  = GAMMA_SWEEP.index(GAMMA_BASE)
        di_d50 = DELTA_SWEEP.index(DELTA_BASE)

        for i, d in enumerate(DELTA_SWEEP):
            means_d.append(rev_grid[i, gi_g3])
            stds_d.append(std_grid[i, gi_g3])

        for j, g in enumerate(GAMMA_SWEEP):
            means_g.append(rev_grid[di_d50, j])
            stds_g.append(std_grid[di_d50, j])


    print("\n>>> Plotting...")

    plot_1d(DELTA_SWEEP, means_d, stds_d,
            xlabel=r'$\delta$ (Sigmoid steepness)',
            title=(r'Sensitivity to $\delta$  ($\gamma=3.0$ fixed, '
                   r'$\tau_k$ fixed, Tasks=80)'),
            chosen_x=int(DELTA_BASE),
            fname='Fig_Sensitivity_Delta_Fixed.png')

    plot_1d(GAMMA_SWEEP, means_g, stds_g,
            xlabel=r'$\gamma$ (convexity exponent)',
            title=(r'Sensitivity to $\gamma$  ($\delta=50$ fixed, '
                   r'$\tau_k$ fixed, Tasks=80)'),
            chosen_x=GAMMA_BASE,
            fname='Fig_Sensitivity_Gamma_Fixed.png')

    plot_heatmap(DELTA_SWEEP, GAMMA_SWEEP, rev_grid,
                 fname='Fig_Heatmap_Fixed.png')

    print(f"\n>>> Done. All figures saved to: {OUTPUT_DIR}/")


    print("\n===== Revenue Grid (rows=δ, cols=γ) =====")
    header = "δ\\γ  " + "  ".join(f"{g:>6.1f}" for g in GAMMA_SWEEP)
    print(header)
    for i, d in enumerate(DELTA_SWEEP):
        row = f"{d:4.0f} " + "  ".join(f"{rev_grid[i,j]:>6.1f}" for j in range(len(GAMMA_SWEEP)))
        print(row)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
