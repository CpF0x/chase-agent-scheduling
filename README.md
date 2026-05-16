# CHASE Task Scheduling Experiments

This repository contains the experiment code for CHASE, a congestion-aware
task scheduling method for heterogeneous edge agents. The code is organized as
a Python package under `src/project` with thin compatibility scripts in
the repository root.

## Repository Layout

```text
src/project/
  config.py                 shared paths and experiment constants
  data.py                   trace loading and environment construction helpers
  models.py                 Agent and Task public imports
  metrics.py                timing, efficiency, and revenue helpers
  algorithms/               CHASE, DyLAN, FL-DRL, and baseline import surfaces
  experiments/              main, time-cost, cross-dataset, and sensitivity runs
  plotting.py               figure export helpers
  tables.py                 CSV table export helpers
  cli.py                    command-line entry point

data/                       place local trace CSV files here
models/                     place local FL-DRL model weights here
outputs/                    generated figures and tables
archive/old_versions/       old scripts and one-off patch utilities
```

Large data files, model weights, caches, and generated figures are intentionally
ignored by Git.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .
```

## Required Local Files

Put the full Borg trace CSV here for the main, time-cost, sensitivity, and
agent-distribution experiments:

```text
data/borg_traces_data.csv
```

Put the Alibaba Cluster Trace v2018 batch-task file here for the cross-dataset
experiment:

```text
data/alibaba_cluster_trace_v2018/batch_task.csv
```

The cross-dataset loader maps the raw Alibaba batch-task table into the local
task model and caches the result here:

```text
data/alibaba_cluster_trace_v2018_mapped.csv
```

If the mapped CSV already exists, the cross-dataset experiment reuses it so the
Alibaba workload mapping stays consistent across runs.

Put the trained FL-DRL model here if running experiments that include FL-DRL:

```text
models/fl_drl_model.pt
```

If the Borg trace is absent, the code falls back to synthetic/default samples
where possible. If the Alibaba batch-task file is absent, the cross-dataset
loader falls back to synthetic Alibaba-like samples. FL-DRL runs require the
model file.

## Running Experiments

Recommended package entry points:

```powershell
python -m project.experiments.main_experiment
python -m project.experiments.timecost
python -m project.experiments.cross_dataset
python -m project.experiments.sensitivity
```

CLI:

```powershell
chase-exp main
chase-exp timecost
chase-exp cross-dataset
chase-exp sensitivity
chase-exp main --quick
chase-exp high-pressure
```

Compatibility scripts still work:

```powershell
python main.py
python main.py --quick
python run_timecost_exp.py
python cross_dataset_experiment.py
python heatmap_sensitivity.py
```

Generated figures are written to `outputs/figures/`; generated CSV tables are
written to `outputs/tables/`.

##  Note

Due to hardware performance differences and other stochastic factors, the exact
numerical results may vary slightly between experimental runs. However, the
relative comparisons and overall performance trends among the algorithms remain
consistent.

## Development Checks

```powershell
python -m py_compile main.py run_timecost_exp.py cross_dataset_experiment.py heatmap_sensitivity.py
python -m pytest
python -m ruff check .
```
