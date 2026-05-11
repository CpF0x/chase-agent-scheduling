"""Generate the main revenue figure."""











import os
import sys

# Import shared experiment module
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import multiprocessing
from functools import partial

from project.config import FIGURES_DIR, TABLES_DIR



import importlib.util, types

_MAIN_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_experiment.py')

def _load_main_module():

    spec = importlib.util.spec_from_file_location('_main_module', _MAIN_PY)
    mod = importlib.util.module_from_spec(spec)

    sys.modules['_main_module'] = mod
    spec.loader.exec_module(mod)
    return mod

# Worker stays module-level for multiprocessing
def _worker(n_task, trial_idx, target_snapshot_task,
            real_trace_data=None, real_agent_dist=None):

    import sys, importlib.util, os
    _mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_experiment.py')
    if '_main_module' not in sys.modules:
        spec = importlib.util.spec_from_file_location('_main_module', _mp)
        mod = importlib.util.module_from_spec(spec)
        sys.modules['_main_module'] = mod
        spec.loader.exec_module(mod)
    m = sys.modules['_main_module']
    return m._worker_full_exp(
        n_task, trial_idx, target_snapshot_task,
        real_trace_data, real_agent_dist
    )


def run_fig1_experiment():


    m = _load_main_module()

    NUM_TRIALS         = m.NUM_TRIALS
    TASK_NUM_RANGE     = m.TASK_NUM_RANGE
    target_snap        = TASK_NUM_RANGE[-1]

    num_workers = min(max(1, multiprocessing.cpu_count() - 2), 8)
    print(f">>> 开始运行 Fig 5-1 实验 (多进程加速，{num_workers} 进程)...")
    print(f"    NUM_TRIALS={NUM_TRIALS}, TASK_NUM_RANGE={TASK_NUM_RANGE}")

    # Load trace data once in the parent process
    print(">>> 正在加载真实数据...")
    global_trace_data = m.load_real_trace_data()
    global_agent_dist = m.load_agent_distribution()
    print("    数据加载完成。")

    results = {
        'tasks': TASK_NUM_RANGE,
        'MCT_rev': [], 'CHASE_rev': [], 'IRS_rev': [], 'NCS_rev': [], 'FL_DRL_rev': [], 'BRTOA_rev': [],
        'MCT_rev_std': [], 'CHASE_rev_std': [], 'IRS_rev_std': [], 'NCS_rev_std': [], 'FL_DRL_rev_std': [], 'BRTOA_rev_std': [],
        'stats': {'main': {}}
    }

    with multiprocessing.Pool(processes=num_workers) as pool:
        for idx, n_task in enumerate(TASK_NUM_RANGE):
            print(f"  [{idx+1}/{len(TASK_NUM_RANGE)}] Tasks={n_task}, 共 {NUM_TRIALS} 次试验...")

            func = partial(
                _worker,
                real_trace_data=global_trace_data,
                real_agent_dist=global_agent_dist,
            )



            func2 = partial(
                _worker,
                n_task,
                target_snapshot_task=target_snap,
                real_trace_data=global_trace_data,
                real_agent_dist=global_agent_dist,
            )
            trial_results = pool.map(func2, range(NUM_TRIALS))

            # Aggregate trial metrics
            temp = {alg: {'revenue':[], 'success_rate':[], 'time':[]}
                    for alg in ['MCT','CHASE','IRS','NCS','FL_DRL','BRTOA']}
            for r in trial_results:
                for alg in temp:
                    temp[alg]['revenue'].append(r[alg]['revenue'])
                    temp[alg]['success_rate'].append(r[alg]['success_rate'])
                    temp[alg]['time'].append(r[alg]['time'])

            results['stats']['main'][n_task] = {}
            for alg in ['MCT','CHASE','IRS','NCS','FL_DRL','BRTOA']:
                arr = np.array(temp[alg]['revenue'])
                results[f'{alg}_rev'].append(float(np.mean(arr)))
                results[f'{alg}_rev_std'].append(float(np.std(arr)))
                results['stats']['main'][n_task][alg] = {
                    'revenue': {
                        'mean':   float(np.mean(arr)),
                        'std':    float(np.std(arr)),
                        'min':    float(np.min(arr)),
                        'max':    float(np.max(arr)),
                        'median': float(np.median(arr)),
                    }
                }
            print(f"      CHASE mean revenue = {results['CHASE_rev'][-1]:.2f}")

    print(">>> 主实验数据收集完毕。")
    return results, m


def plot_fig1(res, m):


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
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
    })

    COLORS = {
        'CHASE': '#E53935',
        'MCT':   '#FB8C00',
        'IRS':   '#26A69A',
        'NCS':   '#78909C',
        'FL_DRL':'#7E57C2',
        'BRTOA': '#00897B',
    }
    MARKERS = {'CHASE':'o','MCT':'s','IRS':'^','NCS':'x','FL_DRL':'d','BRTOA':'p'}
    STYLES  = {
        'CHASE': '-',
        'MCT':   '--',
        'IRS':   '-.',
        'NCS':   ':',
        'FL_DRL':'(0,(3,1,1,1))',
        'BRTOA': (0,(5,2)),
    }
    LABELS = {
        'CHASE':'CHASE (Proposed)', 'MCT':'MCT',
        'IRS':'IRS (Random)', 'NCS':'NCS (Naive)',
        'FL_DRL':'FL-DRL', 'BRTOA':'BRTOA',
    }

    x_data = res['tasks']
    y_keys = [
        ('MCT',    'MCT_rev'),
        ('CHASE',  'CHASE_rev'),
        ('IRS',    'IRS_rev'),
        ('NCS',    'NCS_rev'),
        ('FL_DRL', 'FL_DRL_rev'),
        ('BRTOA',  'BRTOA_rev'),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for alg_key, res_key in y_keys:
        y_data = np.array(res[res_key])
        lw, ms, zorder = (3.5, 10, 10) if alg_key == 'CHASE' else (2.0, 7, 5)

        ls = STYLES[alg_key]
        if isinstance(ls, str) and ls.startswith('('):
            ls = eval(ls)

        ax.plot(x_data, y_data,
                marker=MARKERS[alg_key],
                linestyle=ls,
                color=COLORS[alg_key],
                label=LABELS[alg_key],
                linewidth=lw,
                markersize=ms,
                markeredgecolor='white',
                markeredgewidth=0.8,
                zorder=zorder)

    ax.set_xlabel('Number of Tasks', fontsize=13)
    ax.set_ylabel('Total Revenue', fontsize=13)
    ax.set_title('Total Revenue', fontsize=15, pad=10)

    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc',
              fontsize=10, loc='best')
    ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color='#999999')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(str(FIGURES_DIR), 'Fig_5_1_Revenue.png')
    plt.savefig(path)
    print(f"  [Figure] saved: {path}")

    plt.close()
    print(">>> Fig 5-1 Revenue 图已生成。")


def export_fig1_table(res):

    stats = res.get('stats', {})
    algs = ['MCT','CHASE','IRS','NCS','FL_DRL','BRTOA']
    alg_display = {'MCT':'MCT','CHASE':'CHASE','IRS':'IRS',
                   'NCS':'NCS','FL_DRL':'FL-DRL','BRTOA':'BRTOA'}

    rows = []
    for n_task in sorted(stats.get('main', {}).keys()):
        for alg in algs:
            s = stats['main'][n_task].get(alg, {}).get('revenue', {})
            if not s:
                continue
            rows.append({
                'Tasks':    n_task,
                'Algorithm': alg_display[alg],
                'Mean':   round(s['mean'],   4),
                'Std':    round(s['std'],    4),
                'Min':    round(s['min'],    4),
                'Max':    round(s['max'],    4),
                'Median': round(s['median'], 4),
            })

    if not rows:
        print("  [警告] 无数据，跳过 CSV 导出。")
        return

    df = pd.DataFrame(rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(str(TABLES_DIR), 'Table_Main_Revenue.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  [Table] saved: {path}  ({len(rows)} rows)")

    print(">>> Table_Main_Revenue.csv 已导出。")


# Entry point
def main():
    multiprocessing.freeze_support()
    res, m = run_fig1_experiment()
    plot_fig1(res, m)
    export_fig1_table(res)


if __name__ == '__main__':
    main()
