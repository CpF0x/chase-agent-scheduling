"""Measure algorithm time cost."""



















import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
from functools import partial


# Shared imports

from after_project.config import FIGURES_DIR
from after_project.experiments.main_experiment import (
    CAP_HIGH, CAP_MID, CAP_LOW,
    REAL_TRACE_PATH,
    load_real_trace_data,
    load_agent_distribution,
    Agent, Task,
    create_agents_from_real_data,
    create_tasks_from_real_data,
    run_algorithm,
    get_fl_drl_model,
    FL_DRL_MODEL_PATH,
)


# Experiment settings

EXP_AGENT_COUNT = 40
EXP_TASK_COUNT  = 80
EXP_NUM_TRIALS  = 50

OUTPUT_DIR  = str(FIGURES_DIR)
FIGURE_NAME = "Fig_TimeCost_40agents_80tasks_warmed.png"


# Process-level cache

_PROC_TRACE_DATA  = None
_PROC_AGENT_DIST  = None


def _worker_init(trace_data, agent_dist):








    global _PROC_TRACE_DATA, _PROC_AGENT_DIST


    _PROC_TRACE_DATA = trace_data
    _PROC_AGENT_DIST = agent_dist


    try:
        model = get_fl_drl_model()


        import numpy as _np
        dummy_state = _np.zeros(2 + 2 * EXP_AGENT_COUNT, dtype=_np.float32)
        dummy_mask  = _np.ones(EXP_AGENT_COUNT, dtype=bool)
        model.choose_action(dummy_state, dummy_mask)
        print(f"  [Worker PID={os.getpid()}] 预热完成（FL-DRL 已加载 + JIT 已触发）")
    except Exception as e:
        print(f"  [Worker PID={os.getpid()}] 预热警告: {e}")



# Trial worker

def _timecost_worker(trial_idx):




    global _PROC_TRACE_DATA, _PROC_AGENT_DIST

    random.seed()
    np.random.seed()

    agents = create_agents_from_real_data(EXP_AGENT_COUNT, real_distribution=_PROC_AGENT_DIST)
    tasks  = create_tasks_from_real_data(EXP_TASK_COUNT,   real_data=_PROC_TRACE_DATA)

    trial_times = {}
    for alg in ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']:
        res = run_algorithm(alg, agents, tasks)
        trial_times[alg] = res['time']

    return trial_times





def run_timecost_experiment():
    print(">>> TimeCost 实验开始（公平预热版）")
    print(f"    智能体数 = {EXP_AGENT_COUNT}, 任务数 = {EXP_TASK_COUNT}, 重复次数 = {EXP_NUM_TRIALS}")

    num_workers = min(max(1, multiprocessing.cpu_count() - 2), 8)
    print(f"[*] 启用 {num_workers} 个并行进程")


    print("[*] 主进程加载真实数据集...")
    global_trace_data = load_real_trace_data()
    global_agent_dist = load_agent_distribution()

    print("[*] 创建进程池（initializer 将在每个 worker 中执行预热）...")
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(global_trace_data, global_agent_dist),
    ) as pool:
        print(f"[*] 正在运行 {EXP_NUM_TRIALS} 次 trials（所有进程已预热）...")
        trial_results = pool.map(_timecost_worker, range(EXP_NUM_TRIALS))


    algs = ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']
    time_vals = {}
    time_stds = {}
    print("\n>>> 实验结果汇总（预热后纯算法耗时）：")
    for alg in algs:
        vals = np.array([r[alg] for r in trial_results])
        time_vals[alg] = float(np.mean(vals))
        time_stds[alg] = float(np.std(vals))
        print(f"    {alg:8s}: {time_vals[alg]:.3f} ± {time_stds[alg]:.3f} ms")

    return time_vals, time_stds





def plot_timecost(time_vals, time_stds):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'figure.dpi': 200,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
    })

    algs        = ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']
    bar_labels  = ['MCT', 'CHASE\n(Proposed)', 'IRS\n(Random)', 'DyLAN', 'FL-DRL', 'BRTOA']
    fig3_colors = ['#eaf3e2', '#b4deb6', '#7bc6be', '#439cc4', '#0868a6', '#ebd7b9']

    vals_list = [time_vals[a] for a in algs]
    stds_list = [time_stds[a] for a in algs]
    x_pos     = np.arange(len(algs))
    bar_width  = 0.6

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i, (alg, val, std, color) in enumerate(zip(algs, vals_list, stds_list, fig3_colors)):

        lower_err = min(std, val * 0.8)
        upper_err = std
        asym_err  = [[lower_err], [upper_err]]

        if alg == 'CHASE':
            ax.bar(x_pos[i], val, width=bar_width, color=color,
                   edgecolor='#5bb5b5', linewidth=1.8, linestyle='--',
                   yerr=asym_err, capsize=4,
                   error_kw={'elinewidth': 1.0, 'capthick': 1.0, 'color': '#555555'})
        else:
            ax.bar(x_pos[i], val, width=bar_width, color=color,
                   edgecolor='#888888', linewidth=0.4,
                   yerr=asym_err, capsize=4,
                   error_kw={'elinewidth': 1.0, 'capthick': 1.0, 'color': '#555555'})


    ax.set_yscale('log')


    for i, (val, std) in enumerate(zip(vals_list, stds_list)):
        top = val + std
        if val < 1:
            label = f'{val:.3f}'
        elif val < 100:
            label = f'{val:.1f}'
        else:
            label = f'{val:.0f}'
        ax.annotate(label,
                    xy=(x_pos[i], top),
                    xytext=(0, 6),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#333333')

    ax.set_ylabel('Time (ms, log scale)', fontsize=14)
    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_title(
        f'Algorithmic Time Cost  '
        f'(Agents={EXP_AGENT_COUNT}, Tasks={EXP_TASK_COUNT}, '
        f'{EXP_NUM_TRIALS} Trials, Warmed-up)',
        fontsize=13, pad=12
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=11)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, FIGURE_NAME)
    plt.savefig(out_path)
    plt.close()
    print(f"\n>>> 图表已保存至: {out_path}")





def main():
    multiprocessing.freeze_support()

    time_vals, time_stds = run_timecost_experiment()
    plot_timecost(time_vals, time_stds)
    print(">>> TimeCost 实验完成！")


if __name__ == '__main__':
    main()
