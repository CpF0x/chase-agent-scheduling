import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
import ast
from collections import defaultdict

from project.config import BORG_TRACE_PATH, DATA_DIR, FIGURES_DIR, FL_DRL_MODEL_PATH as DEFAULT_MODEL_PATH, TABLES_DIR


# FL-DRL model cache


FL_DRL_MODEL_PATH = str(DEFAULT_MODEL_PATH)
_FL_DRL_MODEL = None

def get_fl_drl_model():

    global _FL_DRL_MODEL
    if _FL_DRL_MODEL is None:
        from project.algorithms.fldrl import FLDRLInference
        _FL_DRL_MODEL = FLDRLInference(FL_DRL_MODEL_PATH)
    return _FL_DRL_MODEL


# BRTOA baseline

class _LegacyBaseline_BRTOA:
    """Best-response baseline solver."""








    def __init__(self, agents, tasks):
        self.agents = agents
        self.tasks = tasks


        self.max_iterations = 3000


        self.agent_map = {ag.id: ag for ag in agents}

    def calculate_single_task_utility(self, task, agent, current_concurrency, omega=0.0):






        import math
        t_real = calc_real_time(task, agent, current_concurrency, omega)

        effective_deadline = task.deadline - task.tau_k
        x = REVENUE_DELTA * (t_real - effective_deadline)
        sig = 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
        return ((1.0 - sig) ** REVENUE_GAMMA) * task.base_val - REVENUE_COST

    def run(self):





        current_allocation = {t.id: None for t in self.tasks}


        agent_states = {a.id: {'count': 0, 'load_sum': 0.0, 'tasks': []} for a in self.agents}

        iteration = 0
        while iteration < self.max_iterations:
            update_requests = []




            for task in self.tasks:
                current_agent_id = current_allocation[task.id]


                current_utility = 0
                if current_agent_id is not None:
                    curr_ag_count = agent_states[current_agent_id]['count']
                    curr_ag = self.agent_map[current_agent_id]
                    current_utility = self.calculate_single_task_utility(
                        task, curr_ag, curr_ag_count,
                        omega=agent_states[current_agent_id]['load_sum'])


                best_agent_id = current_agent_id
                max_utility = current_utility

                for agent in self.agents:
                    if agent.id == current_agent_id:
                        continue


                    predicted_concurrency = agent_states[agent.id]['count'] + 1


                    task_load = task.wk / agent.capacity
                    if agent_states[agent.id]['load_sum'] + task_load > 1.0:
                        continue



                    predicted_load = agent_states[agent.id]['load_sum'] + task_load
                    utility = self.calculate_single_task_utility(
                        task, agent, predicted_concurrency,
                        omega=predicted_load)


                    if utility > max_utility + 1e-6:
                        max_utility = utility
                        best_agent_id = agent.id


                if best_agent_id != current_agent_id:
                    update_requests.append((task, best_agent_id))


            if not update_requests:

                break



            selected_task, new_agent_id = random.choice(update_requests)



            old_agent_id = current_allocation[selected_task.id]
            if old_agent_id is not None:
                old_ag = self.agent_map[old_agent_id]
                agent_states[old_agent_id]['count'] -= 1
                agent_states[old_agent_id]['load_sum'] -= (selected_task.wk / old_ag.capacity)
                if selected_task in agent_states[old_agent_id]['tasks']:
                    agent_states[old_agent_id]['tasks'].remove(selected_task)


            current_allocation[selected_task.id] = new_agent_id
            if new_agent_id is not None:
                new_ag = self.agent_map[new_agent_id]
                agent_states[new_agent_id]['count'] += 1
                agent_states[new_agent_id]['load_sum'] += (selected_task.wk / new_ag.capacity)
                agent_states[new_agent_id]['tasks'].append(selected_task)

            iteration += 1


        assignment = {ag.id: agent_states[ag.id]['tasks'] for ag in self.agents}
        return assignment



# Global config

NUM_TRIALS = 50
AGENT_COUNT = 40

TASK_NUM_RANGE = [20, 40, 60, 80, 100, 120]



DEADLINE = 0.9
HIGH_PRESSURE_DEADLINE = 0.75
HIGH_PRESSURE_WORKLOAD_SCALE = 1.15
HIGH_PRESSURE_TRIALS = 30
HIGH_PRESSURE_ALGORITHMS = ["CHASE", "BRTOA", "MCT", "IRS", "DyLAN"]








REVENUE_DELTA = 50.0
REVENUE_GAMMA = 3.0
REVENUE_COST  = 5.0


CAP_HIGH = 120
CAP_MID = 70
CAP_LOW = 25


AGENTS_CONFIG = [
    {'type': 'High', 'cap': CAP_HIGH, 'count': 18, 'output': CAP_HIGH*0.5},
    {'type': 'Mid',  'cap': CAP_MID,  'count': 10, 'output': CAP_MID*0.5},
    {'type': 'Low',  'cap': CAP_LOW,  'count': 12, 'output': CAP_LOW*0.5}
]

SKILL_NAMES = ("sense", "preprocess", "infer", "aggregate", "actuate")
SKILL_INDEX = {name: idx for idx, name in enumerate(SKILL_NAMES)}

AGENT_SKILLS_BY_TYPE = {
    "High": ("preprocess", "infer", "aggregate"),
    "Mid": ("sense", "preprocess", "aggregate"),
    "Low": ("sense", "actuate"),
}

SKILL_EFFICIENCY_BY_TYPE = {
    "High": {"preprocess": 0.95, "infer": 1.00, "aggregate": 0.90},
    "Mid": {"sense": 0.90, "preprocess": 0.85, "aggregate": 0.80},
    "Low": {"sense": 0.78, "actuate": 0.82},
}

TASK_TEMPLATES = {
    "Monitoring": ("sense", "aggregate"),
    "Perception": ("sense", "preprocess", "infer"),
    "Control": ("sense", "preprocess", "infer", "actuate"),
    "Fusion": ("sense", "preprocess", "aggregate", "infer"),
}

TASK_TYPE_SEQUENCE = (
    ["Monitoring"] * 5
    + ["Perception"] * 7
    + ["Control"] * 5
    + ["Fusion"] * 3
)

SKILL_WORKLOAD_WEIGHTS = {
    "sense": 0.10,
    "preprocess": 0.20,
    "infer": 0.45,
    "aggregate": 0.15,
    "actuate": 0.10,
}


# Alibaba trace data

REAL_TRACE_PATH = str(BORG_TRACE_PATH)
REAL_TRACE_CACHE = None

def load_real_trace_data(csv_path=REAL_TRACE_PATH, max_rows=50000):













    global REAL_TRACE_CACHE

    if REAL_TRACE_CACHE is not None:
        return REAL_TRACE_CACHE

    print(f"[INFO] Loading real trace data from: {csv_path}")

    try:

        df = pd.read_csv(csv_path, usecols=['average_usage', 'cycles_per_instruction'], nrows=max_rows)


        cpu_usages = []
        for usage_str in df['average_usage'].dropna():
            try:
                usage_dict = ast.literal_eval(usage_str)
                cpu_usage = usage_dict.get('cpus', 0.01)
            except (ValueError, SyntaxError):
                cpu_usage = 0.01
            cpu_usages.append(cpu_usage)


        from scipy import stats
        cpu_array = np.array(cpu_usages)
        percentile_ranks = stats.rankdata(cpu_array, method='average') / len(cpu_array)
        workloads = 25 + percentile_ranks * 55
        workloads = workloads.tolist()


        cpi_series = df['cycles_per_instruction'].dropna()
        cpis = cpi_series.tolist()


        if len(cpis) < len(workloads):
            median_cpi = np.median(cpis) if cpis else 2.0
            cpis = cpis + [median_cpi] * (len(workloads) - len(cpis))

        REAL_TRACE_CACHE = {
            'workloads': workloads,
            'cpis': cpis[:len(workloads)]
        }

        print(f"[OK] Loaded {len(workloads)} real task samples:")
        print(f"     Workload range: [{min(workloads):.1f}, {max(workloads):.1f}], mean={np.mean(workloads):.1f}")
        print(f"     CPI range: [{min(cpis):.2f}, {max(cpis):.2f}], mean={np.mean(cpis):.2f}")

        return REAL_TRACE_CACHE

    except Exception as e:
        print(f"[WARN] Failed to load real trace: {e}, using synthetic data")
        return None


# Alibaba cluster trace workload






ALIBABA_TRACE_CACHE = None
ALIBABA_TRACE_DIR = DATA_DIR / "alibaba_cluster_trace_v2018"
ALIBABA_BATCH_TASK_PATH = ALIBABA_TRACE_DIR / "batch_task.csv"
ALIBABA_MAPPED_TRACE_PATH = DATA_DIR / "alibaba_cluster_trace_v2018_mapped.csv"
ALIBABA_BATCH_TASK_COLUMNS = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]


def _rank_map(values, low, high):
    from scipy import stats
    ranks = stats.rankdata(np.asarray(values, dtype=float), method="average") / len(values)
    return low + ranks * (high - low)


def _build_alibaba_mapped_trace(
    raw_path=ALIBABA_BATCH_TASK_PATH,
    mapped_path=ALIBABA_MAPPED_TRACE_PATH,
    max_rows=50000,
):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Alibaba batch_task.csv not found at {raw_path}. "
            "Download cluster-trace-v2018 batch_task.tar.gz and extract batch_task.csv first."
        )

    rows = []
    for chunk in pd.read_csv(
        raw_path,
        names=ALIBABA_BATCH_TASK_COLUMNS,
        usecols=[
            "task_name",
            "instance_num",
            "task_type",
            "status",
            "start_time",
            "end_time",
            "plan_cpu",
            "plan_mem",
        ],
        chunksize=200000,
    ):
        chunk = chunk[chunk["status"].eq("Terminated")].copy()
        chunk = chunk.dropna(subset=["start_time", "end_time", "plan_cpu", "plan_mem"])
        chunk["duration"] = chunk["end_time"] - chunk["start_time"]
        chunk = chunk[
            (chunk["duration"] > 0)
            & (chunk["plan_cpu"] > 0)
            & (chunk["plan_mem"] > 0)
            & (chunk["instance_num"] > 0)
        ]
        if chunk.empty:
            continue

        rows.append(
            chunk[
                [
                    "task_name",
                    "instance_num",
                    "task_type",
                    "duration",
                    "plan_cpu",
                    "plan_mem",
                ]
            ]
        )
        if sum(len(part) for part in rows) >= max_rows:
            break

    if not rows:
        raise ValueError(f"No usable terminated Alibaba batch tasks found in {raw_path}")

    df = pd.concat(rows, ignore_index=True).head(max_rows)
    cpu_units = df["plan_cpu"].astype(float) / 100.0
    cpu_units = cpu_units.clip(lower=0.01)
    instance_count = df["instance_num"].astype(float).clip(lower=1.0)
    duration = df["duration"].astype(float).clip(lower=1.0)
    mem_units = df["plan_mem"].astype(float).clip(lower=0.01)

    resource_work = cpu_units * duration * np.sqrt(instance_count)
    pressure = duration * mem_units / cpu_units

    mapped = pd.DataFrame(
        {
            "source": "alibaba_cluster_trace_v2018_batch_task",
            "task_name": df["task_name"],
            "task_type": df["task_type"],
            "instance_num": df["instance_num"],
            "duration": duration,
            "plan_cpu": df["plan_cpu"],
            "plan_mem": df["plan_mem"],
            "workload": _rank_map(resource_work, 25.0, 80.0),
            "cpi": _rank_map(pressure, 0.4, 9.0),
        }
    )
    mapped_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(mapped_path, index=False, encoding="utf-8")
    return mapped


def load_alibaba_trace_data(max_rows=50000, rebuild_cache=False):
    global ALIBABA_TRACE_CACHE
    if ALIBABA_TRACE_CACHE is not None and not rebuild_cache:
        return ALIBABA_TRACE_CACHE

    mapped_path = ALIBABA_MAPPED_TRACE_PATH
    if rebuild_cache or not mapped_path.exists():
        print(f"[INFO] Building Alibaba mapped trace from: {ALIBABA_BATCH_TASK_PATH}")
        mapped = _build_alibaba_mapped_trace(
            raw_path=ALIBABA_BATCH_TASK_PATH,
            mapped_path=mapped_path,
            max_rows=max_rows,
        )
    else:
        print(f"[INFO] Loading Alibaba mapped trace from: {mapped_path}")
        mapped = pd.read_csv(mapped_path, usecols=["workload", "cpi"], nrows=max_rows)

    workloads = mapped["workload"].astype(float).tolist()
    cpis = mapped["cpi"].astype(float).tolist()
    ALIBABA_TRACE_CACHE = {"workloads": workloads, "cpis": cpis}
    print(f"[OK] Loaded {len(workloads)} Alibaba task samples:")
    print(f"     Workload range: [{min(workloads):.1f}, {max(workloads):.1f}], mean={np.mean(workloads):.1f}")
    print(f"     CPI range: [{min(cpis):.2f}, {max(cpis):.2f}], mean={np.mean(cpis):.2f}")
    return ALIBABA_TRACE_CACHE

def load_alibaba_synthetic_data(n_samples=50000, seed=42):
    global ALIBABA_TRACE_CACHE
    try:
        return load_alibaba_trace_data(max_rows=n_samples)
    except Exception as e:
        print(f"[WARN] Failed to load real Alibaba trace: {e}, using synthetic Alibaba-like data")
    if ALIBABA_TRACE_CACHE is not None:
        return ALIBABA_TRACE_CACHE

    rng = np.random.RandomState(seed)


    n_light = int(n_samples * 0.80)
    n_heavy = n_samples - n_light
    cpu_light = rng.exponential(scale=12.0, size=n_light)
    cpu_heavy = rng.uniform(30.0, 90.0, size=n_heavy)
    cpu_all = np.concatenate([cpu_light, cpu_heavy])
    rng.shuffle(cpu_all)
    cpu_all = np.clip(cpu_all, 0.5, 100.0)


    from scipy import stats
    pct = stats.rankdata(cpu_all, method='average') / len(cpu_all)
    workloads = (25 + pct * 55).tolist()


    cpis = rng.lognormal(mean=1.0, sigma=0.5, size=n_samples)
    cpis = np.clip(cpis, 0.4, 9.0).tolist()

    ALIBABA_TRACE_CACHE = {'workloads': workloads, 'cpis': cpis}
    print(f"[OK] Synthesized {n_samples} Alibaba-style task samples:")
    print(f"     Workload range: [{min(workloads):.1f}, {max(workloads):.1f}], mean={np.mean(workloads):.1f}")
    print(f"     CPI range: [{min(cpis):.2f}, {max(cpis):.2f}], mean={np.mean(cpis):.2f}")
    return ALIBABA_TRACE_CACHE


# Agent distribution from trace data

AGENT_DISTRIBUTION_CACHE = None

def load_agent_distribution(csv_path=REAL_TRACE_PATH, max_rows=50000):













    global AGENT_DISTRIBUTION_CACHE

    if AGENT_DISTRIBUTION_CACHE is not None:
        return AGENT_DISTRIBUTION_CACHE

    print(f"[INFO] Loading agent distribution from real data...")

    try:
        df = pd.read_csv(csv_path, usecols=['scheduling_class', 'resource_request'], nrows=max_rows)
        df = df.dropna(subset=['scheduling_class', 'resource_request'])


        records = []
        for _, row in df.iterrows():
            sc = int(row['scheduling_class'])
            try:
                rr = ast.literal_eval(row['resource_request'])
                cpu_req = rr.get('cpus', 0.01)
            except:
                cpu_req = 0.01


            if sc == 3 or sc == 2:
                agent_type = 'High'
            elif sc == 1:
                agent_type = 'Mid'
            else:
                agent_type = 'Low'

            records.append({'type': agent_type, 'cpu_req': cpu_req})


        type_counts = {'High': 0, 'Mid': 0, 'Low': 0}
        capacity_samples = {'High': [], 'Mid': [], 'Low': []}

        for r in records:
            type_counts[r['type']] += 1
            capacity_samples[r['type']].append(r['cpu_req'])

        total = sum(type_counts.values())
        type_distribution = {k: v / total for k, v in type_counts.items()}



        capacity_ranges = {
            'High': (90, 120),
            'Mid': (50, 89),
            'Low': (25, 49)
        }

        capacity_mapped = {'High': [], 'Mid': [], 'Low': []}
        for atype in ['High', 'Mid', 'Low']:
            samples = np.array(capacity_samples[atype])
            if len(samples) > 0:

                from scipy import stats
                ranks = stats.rankdata(samples, method='average') / len(samples)
                low, high = capacity_ranges[atype]
                mapped = low + ranks * (high - low)
                capacity_mapped[atype] = mapped.tolist()

        AGENT_DISTRIBUTION_CACHE = {
            'type_distribution': type_distribution,
            'capacity_samples': capacity_mapped
        }

        print(f"[OK] Agent distribution loaded:")
        for atype, pct in type_distribution.items():
            print(f"     {atype}: {pct*100:.1f}%")

        return AGENT_DISTRIBUTION_CACHE

    except Exception as e:
        print(f"[WARN] Failed to load agent distribution: {e}, using default config")
        return None


def create_agents_from_real_data(n_agents=AGENT_COUNT, real_distribution=None):














    if real_distribution is None:
        real_distribution = load_agent_distribution()

    agents = []
    uid = 0

    if real_distribution:
        type_dist = real_distribution['type_distribution']
        capacity_samples = real_distribution['capacity_samples']


        n_high = int(n_agents * type_dist['High'])
        n_low = int(n_agents * type_dist['Low'])
        n_mid = n_agents - n_high - n_low

        type_counts = {'High': n_high, 'Mid': n_mid, 'Low': n_low}

        for atype, count in type_counts.items():
            samples = capacity_samples.get(atype, [])
            for _ in range(count):
                if samples:

                    cap = random.choice(samples)

                    cap = cap + random.uniform(-3, 3)
                    cap = max(CAP_LOW, min(CAP_HIGH, cap))
                else:

                    default_caps = {'High': CAP_HIGH, 'Mid': CAP_MID, 'Low': CAP_LOW}
                    cap = default_caps[atype]

                output = cap * 0.5
                agents.append(Agent(uid, atype, cap, output))
                uid += 1
    else:

        for cfg in AGENTS_CONFIG:
            for _ in range(cfg['count']):
                agents.append(Agent(uid, cfg['type'], cfg['cap'], cfg['output']))
                uid += 1

    return agents

WORKLOAD_RANGE = (25, 80)


# Basic entities


def _default_agent_skills(atype):
    return tuple(AGENT_SKILLS_BY_TYPE.get(atype, ("sense",)))


def _default_skill_efficiency(atype, skills):
    template = SKILL_EFFICIENCY_BY_TYPE.get(atype, {})
    return {skill: float(template.get(skill, 0.65)) for skill in skills}


def _select_task_type(task_id):
    return TASK_TYPE_SEQUENCE[task_id % len(TASK_TYPE_SEQUENCE)]


def _select_required_skills(task_id, workload=None):
    return TASK_TEMPLATES[_select_task_type(task_id)]


def _split_workload_by_skill(total_workload, skills):
    total_weight = sum(SKILL_WORKLOAD_WEIGHTS[skill] for skill in skills)
    pieces = [
        total_workload * SKILL_WORKLOAD_WEIGHTS[skill] / total_weight
        for skill in skills
    ]
    pieces[-1] += total_workload - sum(pieces)
    return pieces


class Subtask:
    """Skill-specific unit inside a complex task."""

    def __init__(self, task_id, index, skill, workload, cpi):
        self.task_id = task_id
        self.id = index
        self.skill = skill
        self.wk = float(workload)
        self.cpi = float(cpi)

    @property
    def key(self):
        return (self.task_id, self.id)


class SubtaskAssignment:
    """Assignment record for one agent-role decision."""

    def __init__(self, task, subtask, agent):
        self.task = task
        self.subtask = subtask
        self.agent = agent

    @property
    def task_id(self):
        return self.task.id

    @property
    def subtask_id(self):
        return self.subtask.id

    @property
    def skill(self):
        return self.subtask.skill

    @property
    def wk(self):
        return self.subtask.wk


class Agent:
    def __init__(self, uid, atype, cap, output, skills=None, skill_efficiency=None):
        self.id = uid
        self.type = atype
        self.capacity = cap
        self.output = output
        self.skills = tuple(skills) if skills is not None else _default_agent_skills(atype)
        self.skill_efficiency = (
            dict(skill_efficiency)
            if skill_efficiency is not None
            else _default_skill_efficiency(atype, self.skills)
        )
        self.current_load = 0.0
        self.concurrent_tasks = []

    def can_execute(self, skill):
        return skill in self.skills

    def reset(self):
        self.current_load = 0.0
        self.concurrent_tasks = []


class Task:
    """Complex task model used by the experiments."""

    DEFAULT_CPI = 2.0

    def __init__(self, uid, real_workload=None, real_cpi=None, subtasks=None, task_type=None):
        self.id = uid
        self.task_type = task_type or _select_task_type(uid)

        if real_workload is not None:
            self.wk = float(real_workload)
        else:
            self.wk = float(random.uniform(*WORKLOAD_RANGE))

        if real_cpi is not None:
            self.cpi = float(real_cpi)
        else:
            self.cpi = Task.DEFAULT_CPI

        if subtasks is None:
            required_skills = TASK_TEMPLATES[self.task_type]
            workloads = _split_workload_by_skill(self.wk, required_skills)
            self.subtasks = [
                Subtask(uid, idx, skill, workloads[idx], self.cpi)
                for idx, skill in enumerate(required_skills)
            ]
        else:
            self.subtasks = list(subtasks)
            self.wk = float(sum(st.wk for st in self.subtasks))

        self.required_skills = tuple(st.skill for st in self.subtasks)
        self.deadline = DEADLINE
        self.base_val = 100.0
        self.tau_k = calc_tau(self.base_val, REVENUE_COST, REVENUE_DELTA, REVENUE_GAMMA)



# CPI efficiency model


CPI_MODEL_PARAMS = {
    'eta_max': 0.75,
    'eta_min': 0.25,
    'cpi_ref': 2.0,
    'k': 1.2,
}

def calc_cpi_efficiency(cpi):















    eta_max = CPI_MODEL_PARAMS['eta_max']
    eta_min = CPI_MODEL_PARAMS['eta_min']
    cpi_ref = CPI_MODEL_PARAMS['cpi_ref']
    k = CPI_MODEL_PARAMS['k']


    efficiency = (eta_max - eta_min) / (1 + np.exp(k * (cpi - cpi_ref))) + eta_min
    return efficiency


def calc_dynamic_output(agent, work_item, skill=None):












    skill = skill or getattr(work_item, "skill", None)
    eta = calc_cpi_efficiency(getattr(work_item, "cpi", Task.DEFAULT_CPI))
    skill_multiplier = agent.skill_efficiency.get(skill, 1.0) if skill else 1.0
    return max(1e-9, agent.capacity * eta * skill_multiplier)



# Performance degradation model



BETA_1 = 0.3
BETA_2 = 0.5
BETA_3 = 0.1


ALPHA_MAP = {
    'High': 0.2,
    'Mid':  0.45,
    'Low':  0.8,
}

def get_alpha(agent):

    return ALPHA_MAP.get(getattr(agent, 'type', 'Mid'), 0.45)

def calc_degradation(omega, concurrency, alpha):











    f = BETA_1 * omega + BETA_2 * omega**2 + BETA_3 * max(0, concurrency - 1)
    return alpha * f

def _representative_skill(task, agent):
    for skill in getattr(task, "required_skills", ()):
        if agent.can_execute(skill):
            return skill
    return getattr(task, "required_skills", (None,))[0]


def calc_real_time(task, agent, concurrency, omega):
    skill = getattr(task, "skill", None)
    if skill is None and hasattr(task, "required_skills"):
        skill = _representative_skill(task, agent)
    base_t = task.wk / calc_dynamic_output(agent, task, skill=skill)
    alpha = get_alpha(agent)
    degradation = calc_degradation(omega, concurrency, alpha)
    return base_t * (1 + degradation)


def calc_subtask_real_time(subtask, agent, concurrency, omega):
    base_t = subtask.wk / calc_dynamic_output(agent, subtask, skill=subtask.skill)
    alpha = get_alpha(agent)
    degradation = calc_degradation(omega, concurrency, alpha)
    return base_t * (1 + degradation)


def assignment_load(record, agent=None):
    target_agent = agent or record.agent
    return record.subtask.wk / target_agent.capacity


def clone_assignment(assignment):
    return {agent_id: list(records) for agent_id, records in assignment.items()}


def task_records_from_assignment(assignment, task):
    return [
        record
        for records in assignment.values()
        for record in records
        if record.task.id == task.id
    ]


def task_records_from_agents(agents, task):
    return [
        record
        for agent in agents
        for record in agent.concurrent_tasks
        if record.task.id == task.id
    ]


def task_is_covered(task, records):
    assigned = {record.subtask.key for record in records if record.task.id == task.id}
    required = {subtask.key for subtask in task.subtasks}
    return required <= assigned


def assigned_agent_ids_for_task(assignment, task):
    return {
        record.agent.id
        for records in assignment.values()
        for record in records
        if record.task.id == task.id
    }


def _agent_load_from_records(agent, records):
    return sum(assignment_load(record, agent) for record in records)


def _projected_assignment(assignment, records_to_add):
    projected = clone_assignment(assignment)
    for record in records_to_add:
        projected.setdefault(record.agent.id, []).append(record)
    return projected


def calc_task_completion_time(task, records, assignment=None):
    if not task_is_covered(task, records):
        return float("inf")

    times = []
    for record in records:
        if record.task.id != task.id:
            continue
        if assignment is not None:
            agent_records = assignment.get(record.agent.id, [])
            concurrency = len(agent_records)
            omega = _agent_load_from_records(record.agent, agent_records)
        else:
            concurrency = len(record.agent.concurrent_tasks)
            omega = record.agent.current_load
        times.append(calc_subtask_real_time(record.subtask, record.agent, concurrency, omega))

    return max(times) if times else float("inf")


def calc_task_revenue(task, records, assignment=None):
    if not task_is_covered(task, records):
        return 0.0
    completion_time = calc_task_completion_time(task, records, assignment=assignment)
    if not np.isfinite(completion_time):
        return 0.0
    return calc_revenue(task, None, 1, completion_time=completion_time)


def _append_record(assignment, record):
    assignment.setdefault(record.agent.id, []).append(record)


def _remove_record(assignment, record):
    records = assignment.get(record.agent.id, [])
    for idx, existing in enumerate(records):
        if existing.task.id == record.task.id and existing.subtask.id == record.subtask.id:
            del records[idx]
            return True
    return False


def remove_task_from_assignment(assignment, task_id):
    removed = []
    for agent_id, records in assignment.items():
        keep = []
        for record in records:
            if record.task.id == task_id:
                removed.append(record)
            else:
                keep.append(record)
        assignment[agent_id] = keep
    return removed


def rebuild_agent_state(agents, assignment):
    for agent in agents:
        agent.reset()
        for record in assignment.get(agent.id, []):
            live_record = SubtaskAssignment(record.task, record.subtask, agent)
            agent.current_load += assignment_load(live_record, agent)
            agent.concurrent_tasks.append(live_record)


def add_live_assignment(agent, task, subtask):
    record = SubtaskAssignment(task, subtask, agent)
    agent.current_load += assignment_load(record, agent)
    agent.concurrent_tasks.append(record)
    return record


def remove_live_records(agents, records):
    keys = {(record.agent.id, record.task.id, record.subtask.id) for record in records}
    for agent in agents:
        kept = []
        for record in agent.concurrent_tasks:
            if (agent.id, record.task.id, record.subtask.id) in keys:
                agent.current_load -= assignment_load(record, agent)
            else:
                kept.append(record)
        agent.concurrent_tasks = kept
        agent.current_load = max(0.0, agent.current_load)


def clone_tasks_for_scenario(tasks, deadline=DEADLINE, workload_scale=1.0):
    """Clone tasks with scaled subtask workloads and a scenario deadline."""
    cloned = []
    for task in tasks:
        subtasks = [
            Subtask(
                task.id,
                subtask.id,
                subtask.skill,
                subtask.wk * workload_scale,
                subtask.cpi,
            )
            for subtask in task.subtasks
        ]
        new_task = Task(task.id, real_cpi=task.cpi, subtasks=subtasks, task_type=task.task_type)
        new_task.deadline = deadline
        cloned.append(new_task)
    return cloned


def create_tasks_from_real_data(n_tasks, real_data=None):










    if real_data is None:
        real_data = load_real_trace_data()

    tasks = []


    if isinstance(real_data, dict) and 'workloads' in real_data:
        workloads = real_data['workloads']
        cpis = real_data.get('cpis', [Task.DEFAULT_CPI] * len(workloads))

        for i in range(n_tasks):
            idx = i % len(workloads)


            base_wk = workloads[idx]
            jitter_wk = random.uniform(-2, 2)
            final_wk = max(25, min(80, base_wk + jitter_wk))


            base_cpi = cpis[idx % len(cpis)]
            jitter_cpi = random.uniform(-0.1, 0.1)
            final_cpi = max(0.4, base_cpi + jitter_cpi)

            tasks.append(Task(i, real_workload=final_wk, real_cpi=final_cpi))
    elif isinstance(real_data, list):

        for i in range(n_tasks):
            idx = i % len(real_data)
            base_wk = real_data[idx]
            jitter = random.uniform(-2, 2)
            final_wk = max(25, min(80, base_wk + jitter))
            tasks.append(Task(i, real_workload=final_wk))
    else:

        tasks = [Task(i) for i in range(n_tasks)]

    return tasks


# Core algorithms


def calc_tau(base_val, cost, delta, gamma):












    import math
    ratio = cost / base_val
    inner = 1.0 - ratio ** (1.0 / gamma)

    inner = max(1e-9, min(1.0 - 1e-9, inner))
    tau_k = (1.0 / delta) * math.log(inner / (1.0 - inner))
    return tau_k

def calc_revenue(task, agent, concurrency, omega=None, completion_time=None):

























    import math
    if omega is None and agent is not None:
        omega = agent.current_load

    real_t = completion_time
    if real_t is None:
        real_t = calc_real_time(task, agent, concurrency, omega)


    effective_deadline = task.deadline - task.tau_k


    x = REVENUE_DELTA * (real_t - effective_deadline)
    sig = 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


    utility = ((1.0 - sig) ** REVENUE_GAMMA) * task.base_val - REVENUE_COST
    return utility

def calc_utility(task, agent, concurrency, omega=None):

    return calc_revenue(task, agent, concurrency, omega)


# Task assignment strategy


def _legacy_tas_find_agent(task, agents, assignment, skip_congestion=False):








    best_agent = None
    best_score = -float('inf')

    for ag in agents:
        omega = task.wk / ag.capacity


        current_tasks = assignment.get(ag.id, [])
        current_load = sum(t.wk / ag.capacity for t in current_tasks)


        if current_load + omega <= 1.0:
            new_conc = len(current_tasks) + 1
            new_total_load = current_load + omega

            if skip_congestion:



                score = ag.capacity - current_load * 10

                if score > best_score:
                    best_agent = ag
                    best_score = score
            else:


                task_utility = calc_utility(task, ag, new_conc, omega=new_total_load)
                if task_utility > 0:

                    will_harm = False
                    for existing_task in current_tasks:
                        if calc_utility(existing_task, ag, new_conc, omega=new_total_load) <= 0:
                            will_harm = True
                            break

                    if not will_harm:

                        real_t = calc_real_time(task, ag, new_conc, new_total_load)
                        time_margin = max(0, (task.deadline - real_t) / task.deadline)


                        conc_penalty = -new_conc * 0.15


                        load_penalty = -current_load * 0.3


                        score = time_margin * 1.5 + conc_penalty + load_penalty

                        if score > best_score:
                            best_agent = ag
                            best_score = score

    return best_agent


CHASE_SCORE_WEIGHTS = {
    "real_time": 1.6,
    "time_margin": 0.7,
    "skill_bonus": 0.08,
    "headroom": 0.03,
    "concurrency_penalty": 0.03,
}

CHASE_REPLACEMENT_CANDIDATE_LIMIT = 12


def _merge_score_weights(overrides=None):
    weights = dict(CHASE_SCORE_WEIGHTS)
    if overrides:
        weights.update(overrides)
    return weights


def _score_subtask_agent(
    subtask,
    agent,
    agent_records,
    skip_congestion=False,
    score_weights=None,
):
    current_load = _agent_load_from_records(agent, agent_records)
    omega = subtask.wk / agent.capacity
    if current_load + omega > 1.0:
        return None

    new_conc = len(agent_records) + 1
    new_total_load = current_load + omega
    if skip_congestion:
        return calc_dynamic_output(agent, subtask, skill=subtask.skill) / max(1.0, agent.capacity)

    real_t = calc_subtask_real_time(subtask, agent, new_conc, new_total_load)
    time_margin = max(0.0, (DEADLINE - real_t) / DEADLINE)
    headroom = 1.0 - new_total_load
    skill_bonus = agent.skill_efficiency.get(subtask.skill, 0.0)
    weights = _merge_score_weights(score_weights)

    return (
        -real_t * weights["real_time"]
        + time_margin * weights["time_margin"]
        + skill_bonus * weights["skill_bonus"]
        + headroom * weights["headroom"]
        - max(0, new_conc - 1) * weights["concurrency_penalty"]
    )


def find_agent_for_subtask(task, subtask, agents, assignment, tentative=None, tabu=None,
                           skip_congestion=False, random_choice=False, score_weights=None,
                           require_skill_match=True):
    tentative = tentative or []
    tabu = tabu or set()
    used_agents = assigned_agent_ids_for_task(assignment, task)
    used_agents.update(record.agent.id for record in tentative if record.task.id == task.id)

    candidates = []
    for agent in agents:
        if agent.id in used_agents or agent.id in tabu:
            continue
        if require_skill_match and not agent.can_execute(subtask.skill):
            continue
        agent_records = list(assignment.get(agent.id, []))
        agent_records.extend(record for record in tentative if record.agent.id == agent.id)
        score = _score_subtask_agent(
            subtask,
            agent,
            agent_records,
            skip_congestion=skip_congestion,
            score_weights=score_weights,
        )
        if score is not None:
            candidates.append((score, agent))

    if not candidates:
        return None
    if random_choice:
        return random.choice(candidates)[1]
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def tas_find_group(task, agents, assignment, skip_congestion=False, tabu=None,
                   random_choice=False, score_weights=None, require_positive_revenue=True,
                   require_skill_match=True, subtask_order="workload_desc"):
    tentative = []
    tabu = tabu or set()
    if subtask_order == "input":
        subtasks = list(task.subtasks)
    elif subtask_order == "workload_asc":
        subtasks = sorted(task.subtasks, key=lambda st: st.wk)
    else:
        subtasks = sorted(task.subtasks, key=lambda st: st.wk, reverse=True)

    for subtask in subtasks:
        agent = find_agent_for_subtask(
            task,
            subtask,
            agents,
            assignment,
            tentative=tentative,
            tabu=tabu,
            skip_congestion=skip_congestion,
            random_choice=random_choice,
            score_weights=score_weights,
            require_skill_match=require_skill_match,
        )
        if agent is None:
            return None
        tentative.append(SubtaskAssignment(task, subtask, agent))

    projected = _projected_assignment(assignment, tentative)
    task_rev = calc_task_revenue(task, tentative, assignment=projected)
    if require_positive_revenue and not skip_congestion and task_rev <= 0:
        return None
    return tentative


def tas_find_single_agent_group(
    task,
    agents,
    assignment,
    skip_congestion=False,
    require_positive_revenue=True,
    score_weights=None,
):
    candidates = []
    for agent in agents:
        if not all(agent.can_execute(subtask.skill) for subtask in task.subtasks):
            continue
        agent_records = list(assignment.get(agent.id, []))
        current_load = _agent_load_from_records(agent, agent_records)
        group = [SubtaskAssignment(task, subtask, agent) for subtask in task.subtasks]
        added_load = sum(assignment_load(record, agent) for record in group)
        if current_load + added_load > 1.0:
            continue

        new_records = agent_records + group
        if skip_congestion:
            score = calc_dynamic_output(agent, task) / max(1.0, agent.capacity)
        else:
            projected = _projected_assignment(assignment, group)
            task_rev = calc_task_revenue(task, group, assignment=projected)
            if require_positive_revenue and task_rev <= 0:
                continue
            completion_time = calc_task_completion_time(task, group, assignment=projected)
            weights = _merge_score_weights(score_weights)
            headroom = 1.0 - _agent_load_from_records(agent, new_records)
            score = (
                -completion_time * weights["real_time"]
                + task_rev * 0.01
                + headroom * weights["headroom"]
                - max(0, len(new_records) - 1) * weights["concurrency_penalty"]
            )
        candidates.append((score, group))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _assignment_agent_stats(assignment):
    loads = {}
    counts = {}
    for agent_id, records in assignment.items():
        counts[agent_id] = len(records)
        loads[agent_id] = sum(assignment_load(record) for record in records)
    return loads, counts


def _agent_stats_after_removing_records(assignment_index, removed_records):
    loads = dict(assignment_index["agent_loads"])
    counts = dict(assignment_index["agent_counts"])
    for record in removed_records:
        agent_id = record.agent.id
        loads[agent_id] = loads.get(agent_id, 0.0) - assignment_load(record)
        counts[agent_id] = counts.get(agent_id, 0) - 1
    return loads, counts


def _score_subtask_agent_from_stats(
    subtask,
    agent,
    current_load,
    current_count,
    skip_congestion=False,
    weights=None,
):
    omega = subtask.wk / agent.capacity
    if current_load + omega > 1.0:
        return None

    new_conc = current_count + 1
    new_total_load = current_load + omega
    if skip_congestion:
        return calc_dynamic_output(agent, subtask, skill=subtask.skill) / max(1.0, agent.capacity)

    real_t = calc_subtask_real_time(subtask, agent, new_conc, new_total_load)
    time_margin = max(0.0, (DEADLINE - real_t) / DEADLINE)
    headroom = 1.0 - new_total_load
    skill_bonus = agent.skill_efficiency.get(subtask.skill, 0.0)

    return (
        -real_t * weights["real_time"]
        + time_margin * weights["time_margin"]
        + skill_bonus * weights["skill_bonus"]
        + headroom * weights["headroom"]
        - max(0, new_conc - 1) * weights["concurrency_penalty"]
    )


def _tas_find_group_chase_cached(
    task,
    agents,
    assignment,
    skip_congestion=False,
    subtask_order="workload_desc",
    require_positive_revenue=True,
    require_skill_match=True,
    score_weights=None,
    agent_stats=None,
):
    tentative = []
    used_agents = assigned_agent_ids_for_task(assignment, task)
    if agent_stats is None:
        agent_loads, agent_counts = _assignment_agent_stats(assignment)
    else:
        agent_loads, agent_counts = (dict(agent_stats[0]), dict(agent_stats[1]))
    weights = _merge_score_weights(score_weights)

    if subtask_order == "input":
        subtasks = list(task.subtasks)
    elif subtask_order == "workload_asc":
        subtasks = sorted(task.subtasks, key=lambda st: st.wk)
    else:
        subtasks = sorted(task.subtasks, key=lambda st: st.wk, reverse=True)

    for subtask in subtasks:
        best_score = -float("inf")
        best_agent = None
        for agent in agents:
            if agent.id in used_agents:
                continue
            if require_skill_match and not agent.can_execute(subtask.skill):
                continue
            score = _score_subtask_agent_from_stats(
                subtask,
                agent,
                agent_loads.get(agent.id, 0.0),
                agent_counts.get(agent.id, 0),
                skip_congestion=skip_congestion,
                weights=weights,
            )
            if score is not None and score > best_score:
                best_score = score
                best_agent = agent

        if best_agent is None:
            return None

        record = SubtaskAssignment(task, subtask, best_agent)
        tentative.append(record)
        used_agents.add(best_agent.id)
        agent_counts[best_agent.id] = agent_counts.get(best_agent.id, 0) + 1
        agent_loads[best_agent.id] = agent_loads.get(best_agent.id, 0.0) + assignment_load(
            record,
            best_agent,
        )

    projected = _projected_assignment(assignment, tentative)
    task_rev = calc_task_revenue(task, tentative, assignment=projected)
    if require_positive_revenue and not skip_congestion and task_rev <= 0:
        return None
    return tentative


def _find_chase_group(
    task,
    agents,
    assignment,
    skip_congestion=False,
    subtask_order="workload_desc",
    require_positive_revenue=True,
    require_skill_match=True,
    score_weights=None,
    single_agent_tasks=False,
    agent_stats=None,
):
    if single_agent_tasks:
        return tas_find_single_agent_group(
            task,
            agents,
            assignment,
            skip_congestion=skip_congestion,
            require_positive_revenue=require_positive_revenue,
            score_weights=score_weights,
        )
    return _tas_find_group_chase_cached(
        task,
        agents,
        assignment,
        skip_congestion=skip_congestion,
        subtask_order=subtask_order,
        require_positive_revenue=require_positive_revenue,
        require_skill_match=require_skill_match,
        score_weights=score_weights,
        agent_stats=agent_stats,
    )


def commit_group_assignment(assignment, group):
    for record in group:
        _append_record(assignment, record)


def tas_find_agent(task, agents, assignment, skip_congestion=False):
    group = tas_find_group(task, agents, assignment, skip_congestion=skip_congestion)
    return group[0].agent if group else None


# DyLAN baseline


class _LegacyDyLAN:
    """DyLAN baseline implementation."""
















    def __init__(self, agents, tasks,
                 T_max=5, top_k=None, consistency_theta=2/3):








        self.agents = list(agents)
        self.tasks  = list(tasks)
        self.T_max  = T_max
        self.top_k  = top_k if top_k is not None else max(1, len(agents) // 2)
        self.theta  = consistency_theta


        self.V_layers = []
        self.E_layers = []

        self.messages = []




    def _llm_ranker(self, active_agents, t):











        scored = []
        for ag in active_agents:

            tasks_on_ag = ag.concurrent_tasks
            n = len(tasks_on_ag)
            if n == 0:
                score = 0.0
            else:
                score = sum(
                    calc_revenue(t_k, ag, n,
                                 omega=ag.current_load)
                    for t_k in tasks_on_ag
                )
            scored.append((score, ag))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored




    def _check_consistency(self, active_agents):








        if not active_agents:
            return True
        loads = [ag.current_load for ag in active_agents]
        mean_load = sum(loads) / len(loads)

        consistent = sum(1 for l in loads if abs(l - mean_load) < 0.10)
        return (consistent / len(active_agents)) >= self.theta




    def _forward_message_passing(self, active_agents, prev_assignment):










        for ag in active_agents:
            ag.reset()


        sorted_tasks = sorted(self.tasks, key=lambda t: t.wk)
        assignment = {ag.id: [] for ag in active_agents}

        for task in sorted_tasks:
            best_ag    = None
            best_score = -float('inf')
            for ag in active_agents:
                omega = task.wk / ag.capacity
                if ag.current_load + omega <= 1.0:
                    n     = len(ag.concurrent_tasks) + 1
                    score = calc_revenue(task, ag, n,
                                        omega=ag.current_load + omega)
                    if score > best_score:
                        best_ag    = ag
                        best_score = score
            if best_ag:
                best_ag.current_load += task.wk / best_ag.capacity
                best_ag.concurrent_tasks.append(task)
                assignment[best_ag.id].append(task)


        messages = {}
        for ag in active_agents:
            n = len(ag.concurrent_tasks)
            if n == 0:
                rev = 0.0
            else:
                rev = sum(
                    calc_revenue(tk, ag, n, omega=ag.current_load)
                    for tk in ag.concurrent_tasks
                )
            messages[ag.id] = {'revenue': rev, 'assignment': assignment[ag.id]}
        return messages




    def run_task_solving(self):








        t = 1
        active_agents = list(self.agents)
        self.V_layers  = [list(active_agents)]
        self.messages  = [{}]
        stop_flag = False
        prev_assignment = {}

        while t <= self.T_max and not stop_flag:

            msgs_t = self._forward_message_passing(active_agents, prev_assignment)
            self.messages.append(msgs_t)


            ranked = self._llm_ranker(active_agents, t)
            top_agents = [ag for (_, ag) in ranked[:self.top_k]]


            new_agent_ids = {ag.id for ag in top_agents}
            edges = {(ag.id, nag.id)
                     for ag in active_agents
                     for nag in top_agents}
            self.E_layers.append(edges)

            self.V_layers.append(list(top_agents))


            if self._check_consistency(active_agents):
                stop_flag = True

            prev_assignment = {ag.id: msgs_t.get(ag.id, {}).get('assignment', [])
                               for ag in active_agents}
            active_agents = top_agents
            t += 1




        for ag in self.agents:
            ag.reset()


        last_t = min(t - 1, len(self.messages) - 1)
        if last_t > 0:
            last_msgs = self.messages[last_t]
            for ag in active_agents:
                assigned_tasks = last_msgs.get(ag.id, {}).get('assignment', [])
                for tk in assigned_tasks:
                    ag.current_load += tk.wk / ag.capacity
                    ag.concurrent_tasks.append(tk)
        else:

            sorted_tasks = sorted(self.tasks, key=lambda tk: tk.wk)
            for task in sorted_tasks:
                best_ag, best_score = None, -float('inf')
                for ag in self.agents:
                    omega = task.wk / ag.capacity
                    if ag.current_load + omega <= 1.0:
                        n     = len(ag.concurrent_tasks) + 1
                        score = calc_revenue(task, ag, n,
                                             omega=ag.current_load + omega)
                        if score > best_score:
                            best_ag, best_score = ag, score
                if best_ag:
                    best_ag.current_load += task.wk / best_ag.capacity
                    best_ag.concurrent_tasks.append(task)

        return {ag.id: ag.concurrent_tasks for ag in self.agents}




    def run_team_optimization(self, k=None):











        if k is None:
            k = self.top_k

        T = len(self.V_layers) - 1
        if T <= 0:

            return self.agents, {ag.id: 0.0 for ag in self.agents}


        I = {ag.id: 0.0 for ag in self.agents}

        last_layer = self.V_layers[T]
        if last_layer:

            positive_count = sum(
                1 for ag in last_layer
                if sum(calc_revenue(tk, ag, len(ag.concurrent_tasks),
                                    omega=ag.current_load)
                       for tk in ag.concurrent_tasks) > 0
            )
            if positive_count == 0:
                positive_count = len(last_layer)
            for ag in last_layer:
                I[ag.id] = 1.0 / positive_count


        for t in range(T, 1, -1):
            layer_t   = self.V_layers[t]
            layer_t1  = self.V_layers[t - 1]
            edges     = self.E_layers[t - 1] if (t - 1) < len(self.E_layers) else set()
            msgs_t    = self.messages[t] if t < len(self.messages) else {}
            msgs_t1   = self.messages[t - 1] if (t - 1) < len(self.messages) else {}

            for ag_i in layer_t:

                rev_i = msgs_t.get(ag_i.id, {}).get('revenue', 0.0)


                predecessors = [
                    ag_j for ag_j in layer_t1
                    if (ag_j.id, ag_i.id) in edges
                ]
                if not predecessors:
                    continue


                pred_revs = []
                for ag_j in predecessors:
                    rev_j = msgs_t1.get(ag_j.id, {}).get('revenue', 0.0)
                    pred_revs.append(max(0.0, rev_j))

                total_pred_rev = sum(pred_revs) + 1e-12

                for ag_j, rev_j in zip(predecessors, pred_revs):
                    w_ji = rev_j / total_pred_rev

                    I[ag_j.id] = I.get(ag_j.id, 0.0) + I.get(ag_i.id, 0.0) * w_ji



        sorted_agents = sorted(self.agents,
                               key=lambda ag: I.get(ag.id, 0.0),
                               reverse=True)
        A_opt = sorted_agents[:k]

        return A_opt, I



# CHASE algorithm

def _legacy_run_CHASE_full(agents, tasks):












    assignment = {ag.id: [] for ag in agents}

    task_agent = {}


    for task in tasks:
        best_ag = tas_find_agent(task, agents, assignment)
        if best_ag:
            assignment[best_ag.id].append(task)
            task_agent[task.id] = best_ag


    conflict_set = set()
    agents_to_check = set(ag.id for ag in agents)


    agent_map = {ag.id: ag for ag in agents}


    max_iterations = 100
    iteration = 0

    while (agents_to_check or conflict_set) and iteration < max_iterations:
        iteration += 1


        for ag_id in list(agents_to_check):
            ag = agent_map[ag_id]
            P_req = assignment[ag_id]

            if len(P_req) > 1:

                L_total = sum(t.wk / ag.capacity for t in P_req)
                N_i = len(P_req)


                U_joint_conc = sum(
                    calc_utility(t, ag, N_i, omega=L_total) for t in P_req
                )


                U_max_excl = max(
                    calc_utility(t, ag, 1, omega=0.0) for t in P_req
                )


                if L_total <= 1.0 and U_joint_conc > U_max_excl:

                    pass
                else:

                    conflict_set.add(ag_id)


        agents_to_check.clear()


        for ag_id in list(conflict_set):
            ag = agent_map[ag_id]
            P_curr = list(assignment[ag_id])

            while True:

                L_curr = sum(t.wk / ag.capacity for t in P_curr)
                N_curr = len(P_curr)

                if N_curr == 0:
                    break


                U_joint_curr = sum(
                    calc_utility(t, ag, N_curr, omega=L_curr) for t in P_curr
                )
                U_max_excl = max(
                    calc_utility(t, ag, 1, omega=0.0) for t in P_curr
                )


                if N_curr == 1 or (L_curr <= 1.0 and U_joint_curr > U_max_excl):
                    break




                alt_costs = {}
                for t in P_curr:

                    temp_assignment = {
                        aid: [task for task in tlist if task.id != t.id]
                        for aid, tlist in assignment.items()
                    }
                    alt_agent = tas_find_agent(t, agents, temp_assignment)

                    if alt_agent:

                        alt_costs[t.id] = t.wk
                    else:

                        alt_costs[t.id] = float('inf')


                p_drop = min(P_curr, key=lambda t: alt_costs[t.id])


                P_curr.remove(p_drop)
                assignment[ag_id].remove(p_drop)

                if p_drop.id in task_agent:
                    del task_agent[p_drop.id]


                new_agent = tas_find_agent(p_drop, agents, assignment)
                if new_agent:
                    assignment[new_agent.id].append(p_drop)
                    task_agent[p_drop.id] = new_agent

                    agents_to_check.add(new_agent.id)


                agents_to_check.add(ag_id)


        conflict_set.clear()



    for ag in agents:
        ag.reset()
        for task in assignment[ag.id]:
            ag.current_load += task.wk / ag.capacity
            ag.concurrent_tasks.append(task)

    return assignment


# CHASE ablation variants

def _legacy_run_CHASE_ablation(agents, tasks, skip_congestion=False, skip_repick=False, random_drop=False):













    assignment = {ag.id: [] for ag in agents}
    task_agent = {}


    for task in tasks:
        best_ag = tas_find_agent(task, agents, assignment, skip_congestion=True)
        if best_ag:
            assignment[best_ag.id].append(task)
            task_agent[task.id] = best_ag


    if skip_congestion:

        for ag in agents:
            ag.reset()
            for task in assignment[ag.id]:
                ag.current_load += task.wk / ag.capacity
                ag.concurrent_tasks.append(task)
        return assignment


    conflict_set = set()
    agents_to_check = set(ag.id for ag in agents)
    agent_map = {ag.id: ag for ag in agents}

    max_iterations = 100
    iteration = 0

    while (agents_to_check or conflict_set) and iteration < max_iterations:
        iteration += 1


        for ag_id in list(agents_to_check):
            ag = agent_map[ag_id]
            P_req = assignment[ag_id]

            if len(P_req) > 1:
                L_total = sum(t.wk / ag.capacity for t in P_req)
                N_i = len(P_req)

                U_joint_conc = sum(calc_utility(t, ag, N_i, omega=L_total) for t in P_req)
                U_max_excl = max(calc_utility(t, ag, 1, omega=0.0) for t in P_req)


                is_safe = L_total <= 1.0 and U_joint_conc > U_max_excl

                if not is_safe:
                    conflict_set.add(ag_id)

        agents_to_check.clear()


        for ag_id in list(conflict_set):
            ag = agent_map[ag_id]
            P_curr = list(assignment[ag_id])

            while True:
                L_curr = sum(t.wk / ag.capacity for t in P_curr)
                N_curr = len(P_curr)

                if N_curr == 0:
                    break

                U_joint_curr = sum(calc_utility(t, ag, N_curr, omega=L_curr) for t in P_curr)
                U_max_excl = max(calc_utility(t, ag, 1, omega=0.0) for t in P_curr)


                should_stop = N_curr == 1 or (L_curr <= 1.0 and U_joint_curr > U_max_excl)

                if should_stop:
                    break


                if random_drop:

                    p_drop = random.choice(P_curr)
                else:

                    alt_costs = {}
                    for t in P_curr:
                        temp_assignment = {
                            aid: [task for task in tlist if task.id != t.id]
                            for aid, tlist in assignment.items()
                        }

                        alt_agent = tas_find_agent(t, agents, temp_assignment, skip_congestion=False)

                        if alt_agent:
                            alt_costs[t.id] = t.wk
                        else:
                            alt_costs[t.id] = float('inf')

                    p_drop = min(P_curr, key=lambda t: alt_costs[t.id])


                P_curr.remove(p_drop)
                assignment[ag_id].remove(p_drop)

                if p_drop.id in task_agent:
                    del task_agent[p_drop.id]


                if not skip_repick:

                    new_agent = tas_find_agent(p_drop, agents, assignment, skip_congestion=False)
                    if new_agent:
                        assignment[new_agent.id].append(p_drop)
                        task_agent[p_drop.id] = new_agent
                        agents_to_check.add(new_agent.id)

                agents_to_check.add(ag_id)

        conflict_set.clear()


    for ag in agents:
        ag.reset()
        for task in assignment[ag.id]:
            ag.current_load += task.wk / ag.capacity
            ag.concurrent_tasks.append(task)

    return assignment

def _legacy_run_algorithm(mode, agents, tasks):
    for ag in agents: ag.reset()
    start_time = time.perf_counter()
    total_revenue = 0
    success_count = 0


    if mode == 'IRS':
        irs_tasks = list(tasks)
        random.shuffle(irs_tasks)

        for task in irs_tasks:
            feas_agents = []
            for ag in agents:
                omega = task.wk / ag.capacity
                if ag.current_load + omega <= 1.0:
                    feas_agents.append(ag)

            if feas_agents:
                target_ag = random.choice(feas_agents)
                target_ag.current_load += (task.wk / target_ag.capacity)
                target_ag.concurrent_tasks.append(task)


    elif mode == 'CHASE':

        run_CHASE_full(agents, tasks)


    elif mode == 'CHASE_NO_CONG':

        run_CHASE_ablation(agents, tasks, skip_congestion=True)
    elif mode == 'CHASE_NO_REPICK':

        run_CHASE_ablation(agents, tasks, skip_repick=True)
    elif mode == 'CHASE_NO_SMART':

        run_CHASE_ablation(agents, tasks, random_drop=True)




    elif mode == 'FL_DRL':

        fl_drl_model = get_fl_drl_model()


        sorted_tasks = sorted(tasks, key=lambda t: t.wk, reverse=False)

        for task in sorted_tasks:

            state = fl_drl_model.build_state(task, agents)


            feasible_mask = np.zeros(len(agents), dtype=bool)
            for i, ag in enumerate(agents):
                omega = task.wk / ag.capacity
                if ag.current_load + omega <= 1.0:
                    feasible_mask[i] = True


            selected_idx = fl_drl_model.choose_action(state, feasible_mask)


            if selected_idx is not None:
                target_ag = agents[selected_idx]
                target_ag.current_load += (task.wk / target_ag.capacity)
                target_ag.concurrent_tasks.append(task)


    elif mode == 'BRTOA':
        brtoa_solver = Baseline_BRTOA(agents, tasks)
        assignment = brtoa_solver.run()


        for ag in agents:
            ag.reset()
            for task in assignment.get(ag.id, []):
                ag.current_load += task.wk / ag.capacity
                ag.concurrent_tasks.append(task)


    elif mode == 'MCT':
        sorted_tasks = sorted(tasks, key=lambda t: t.wk, reverse=False)
        for task in sorted_tasks:
            best_agent = None
            best_score = -float('inf')
            for ag in agents:
                omega = task.wk / ag.capacity
                if ag.current_load + omega <= 1.0:
                    n = len(ag.concurrent_tasks) + 1
                    est_time = calc_real_time(task, ag, n, ag.current_load)
                    score = -est_time
                    if score > best_score:
                        best_agent = ag
                        best_score = score
            if best_agent:
                best_agent.current_load += (task.wk / best_agent.capacity)
                best_agent.concurrent_tasks.append(task)


    elif mode == 'DyLAN':
        dylan_solver = DyLAN(agents, tasks)
        dylan_solver.run_task_solving()


    else:
        raise ValueError(f'Unknown algorithm mode: {mode}')


    for ag in agents:
        n = len(ag.concurrent_tasks)
        if n > 0:
            for t in ag.concurrent_tasks:
                rev = calc_revenue(t, ag, n)
                total_revenue += rev
                if rev > 0: success_count += 1

    end_time = (time.perf_counter() - start_time) * 1000

    utilization = {'High': [], 'Mid': [], 'Low': []}
    concurrency_counts = {'High': [], 'Mid': [], 'Low': []}

    for ag in agents:
        utilization[ag.type].append(ag.current_load)
        concurrency_counts[ag.type].append(len(ag.concurrent_tasks))

    avg_util = {k: np.mean(v) if v else 0 for k, v in utilization.items()}
    avg_conc = {k: np.mean(v) if v else 0 for k, v in concurrency_counts.items()}

    return {
        'revenue': total_revenue,
        'success_rate': success_count / len(tasks) if len(tasks) > 0 else 0,
        'time': end_time,
        'utilization': avg_util,
        'concurrency': avg_conc
    }


def _empty_assignment(agents):
    return {agent.id: [] for agent in agents}


def _unique_tasks(records):
    tasks_by_id = {}
    for record in records:
        tasks_by_id[record.task.id] = record.task
    return list(tasks_by_id.values())


def _task_revenue_from_assignment(task, assignment):
    return calc_task_revenue(task, task_records_from_assignment(assignment, task), assignment)


def _build_assignment_index(assignment):
    records_by_task = {}
    tasks_by_agent = {}
    agent_loads = {}
    agent_counts = {}
    assigned_tasks_by_id = {}

    for agent_id, records in assignment.items():
        agent_counts[agent_id] = len(records)
        agent_loads[agent_id] = sum(assignment_load(record) for record in records)
        agent_tasks = {}
        for record in records:
            records_by_task.setdefault(record.task.id, []).append(record)
            agent_tasks[record.task.id] = record.task
            if record.task.id not in assigned_tasks_by_id:
                assigned_tasks_by_id[record.task.id] = record.task
        tasks_by_agent[agent_id] = agent_tasks

    return {
        "records_by_task": records_by_task,
        "tasks_by_agent": tasks_by_agent,
        "agent_loads": agent_loads,
        "agent_counts": agent_counts,
        "assigned_tasks": list(assigned_tasks_by_id.values()),
        "task_revenue": {},
    }


def _task_revenue_from_index(task, assignment, assignment_index=None):
    if assignment_index is None:
        return _task_revenue_from_assignment(task, assignment)
    task_revenue = assignment_index["task_revenue"]
    if task.id not in task_revenue:
        records = assignment_index["records_by_task"].get(task.id, [])
        task_revenue[task.id] = calc_task_revenue(task, records, assignment)
    return task_revenue[task.id]


def _agent_joint_revenue(agent_id, assignment):
    records = assignment.get(agent_id, [])
    return sum(_task_revenue_from_assignment(task, assignment) for task in _unique_tasks(records))


def _agent_exclusive_benchmark(agent_id, assignment):
    records = assignment.get(agent_id, [])
    if not records:
        return 0.0

    values = []
    for record in records:
        temp = clone_assignment(assignment)
        temp[agent_id] = [record]
        values.append(_task_revenue_from_assignment(record.task, temp))
    return max(values) if values else 0.0


def _agent_load(agent_id, agent, assignment):
    return _agent_load_from_records(agent, assignment.get(agent_id, []))


def _agent_is_safe(agent_id, agent, assignment):
    records = assignment.get(agent_id, [])
    if len(records) <= 1:
        return True
    load = _agent_load(agent_id, agent, assignment)
    return load <= 1.0 and _agent_joint_revenue(agent_id, assignment) > _agent_exclusive_benchmark(
        agent_id, assignment
    )


def _record_marginal_value(record, assignment):
    involved_tasks = _unique_tasks(assignment.get(record.agent.id, []))
    current = sum(_task_revenue_from_assignment(task, assignment) for task in involved_tasks)
    temp = clone_assignment(assignment)
    _remove_record(temp, record)
    after = sum(_task_revenue_from_assignment(task, temp) for task in involved_tasks)
    return current - after


def _record_drop_cost(
    record,
    agents,
    assignment,
    tabu,
    score_weights=None,
    require_skill_match=True,
    assignment_index=None,
):
    involved_tasks = (
        list(assignment_index["tasks_by_agent"].get(record.agent.id, {}).values())
        if assignment_index is not None
        else _unique_tasks(assignment.get(record.agent.id, []))
    )
    if record.task not in involved_tasks:
        involved_tasks.append(record.task)

    current = sum(
        _task_revenue_from_index(task, assignment, assignment_index)
        for task in involved_tasks
    )
    temp = clone_assignment(assignment)
    _remove_record(temp, record)

    task_tabu = set(tabu[record.task.id])
    task_tabu.add(record.agent.id)
    replacement_agent = find_agent_for_subtask(
        record.task,
        record.subtask,
        agents,
        temp,
        tabu=task_tabu,
        skip_congestion=False,
        score_weights=score_weights,
        require_skill_match=require_skill_match,
    )

    if replacement_agent is None:
        remove_task_from_assignment(temp, record.task.id)
    else:
        replacement = SubtaskAssignment(record.task, record.subtask, replacement_agent)
        _append_record(temp, replacement)

    temp_index = _build_assignment_index(temp)
    after = sum(_task_revenue_from_index(task, temp, temp_index) for task in involved_tasks)
    return current - after


def _replace_or_remove_record(
    record,
    agents,
    assignment,
    tabu,
    skip_repick=False,
    score_weights=None,
    require_skill_match=True,
):
    _remove_record(assignment, record)
    tabu[record.task.id].add(record.agent.id)

    if skip_repick:
        removed = remove_task_from_assignment(assignment, record.task.id)
        return removed, set()

    replacement_agent = find_agent_for_subtask(
        record.task,
        record.subtask,
        agents,
        assignment,
        tabu=tabu[record.task.id],
        skip_congestion=False,
        score_weights=score_weights,
        require_skill_match=require_skill_match,
    )
    if replacement_agent is None:
        removed = remove_task_from_assignment(assignment, record.task.id)
        return removed + [record], {record.agent.id}

    replacement = SubtaskAssignment(record.task, record.subtask, replacement_agent)
    _append_record(assignment, replacement)
    return [record], {record.agent.id, replacement_agent.id}


def _ordered_chase_tasks(tasks, task_order):
    if task_order == "input":
        return list(tasks)
    if task_order == "workload_desc":
        return sorted(tasks, key=lambda task: task.wk, reverse=True)
    return sorted(tasks, key=lambda task: task.wk)


def _assignment_total_revenue(tasks, assignment):
    return sum(_task_revenue_from_assignment(task, assignment) for task in tasks)


def _tasks_for_agent_ids(assignment, agent_ids, assignment_index=None):
    if assignment_index is not None:
        tasks_by_id = {}
        for agent_id in agent_ids:
            tasks_by_id.update(assignment_index["tasks_by_agent"].get(agent_id, {}))
        return tasks_by_id

    tasks_by_id = {}
    for agent_id in agent_ids:
        for record in assignment.get(agent_id, []):
            tasks_by_id[record.task.id] = record.task
    return tasks_by_id


def _replacement_affected_tasks(
    current_assignment,
    projected_assignment,
    removed_records,
    added_records,
    current_index=None,
    projected_index=None,
):
    affected_agent_ids = {
        record.agent.id
        for record in removed_records
    }
    affected_agent_ids.update(record.agent.id for record in added_records)

    tasks_by_id = {}
    tasks_by_id.update(
        _tasks_for_agent_ids(current_assignment, affected_agent_ids, current_index)
    )
    tasks_by_id.update(
        _tasks_for_agent_ids(projected_assignment, affected_agent_ids, projected_index)
    )

    for record in removed_records:
        tasks_by_id[record.task.id] = record.task
    for record in added_records:
        tasks_by_id[record.task.id] = record.task

    return tasks_by_id.values()


def _replacement_delta(
    current_assignment,
    projected_assignment,
    removed_records,
    added_records,
    current_index=None,
    projected_index=None,
):
    affected_tasks = _replacement_affected_tasks(
        current_assignment,
        projected_assignment,
        removed_records,
        added_records,
        current_index=current_index,
        projected_index=projected_index,
    )
    current_value = sum(
        _task_revenue_from_index(task, current_assignment, current_index)
        for task in affected_tasks
    )
    projected_value = sum(
        _task_revenue_from_index(task, projected_assignment, projected_index)
        for task in affected_tasks
    )
    return projected_value - current_value


def _try_value_aware_replacement(
    task,
    agents,
    tasks,
    assignment,
    init_skip_congestion=False,
    subtask_order="workload_desc",
    require_positive_revenue=True,
    require_skill_match=True,
    score_weights=None,
    single_agent_tasks=False,
):
    current_index = _build_assignment_index(assignment)
    assigned_tasks = current_index["assigned_tasks"]
    candidate_scores = [
        (
            _task_revenue_from_index(assigned_task, assignment, current_index),
            assigned_task,
        )
        for assigned_task in assigned_tasks
    ]
    candidates = [
        assigned_task
        for _, assigned_task in sorted(candidate_scores, key=lambda item: item[0])
    ][:CHASE_REPLACEMENT_CANDIDATE_LIMIT]

    best_delta = 0.0
    best_assignment = None
    for victim in candidates:
        trial_assignment = clone_assignment(assignment)
        removed_records = remove_task_from_assignment(trial_assignment, victim.id)
        agent_stats = _agent_stats_after_removing_records(current_index, removed_records)
        group = _find_chase_group(
            task,
            agents,
            trial_assignment,
            skip_congestion=init_skip_congestion,
            subtask_order=subtask_order,
            require_positive_revenue=require_positive_revenue,
            require_skill_match=require_skill_match,
            score_weights=score_weights,
            single_agent_tasks=single_agent_tasks,
            agent_stats=agent_stats,
        )
        if not group:
            continue

        projected = clone_assignment(trial_assignment)
        commit_group_assignment(projected, group)
        delta = _replacement_delta(
            assignment,
            projected,
            removed_records,
            group,
            current_index=current_index,
        )
        if delta > best_delta:
            best_delta = delta
            best_assignment = projected

    if best_assignment is None:
        return False

    assignment.clear()
    assignment.update(best_assignment)
    return True


def _run_chase_group(
    agents,
    tasks,
    skip_congestion=False,
    skip_repick=False,
    random_drop=False,
    init_skip_congestion=None,
    task_order="workload_asc",
    subtask_order="workload_desc",
    require_positive_revenue=True,
    require_skill_match=True,
    score_weights=None,
    use_value_replacement=False,
    single_agent_tasks=False,
):
    assignment = _empty_assignment(agents)
    tabu = defaultdict(set)
    if init_skip_congestion is None:
        init_skip_congestion = skip_congestion

    for task in _ordered_chase_tasks(tasks, task_order):
        group = _find_chase_group(
            task,
            agents,
            assignment,
            skip_congestion=init_skip_congestion,
            subtask_order=subtask_order,
            require_positive_revenue=require_positive_revenue,
            require_skill_match=require_skill_match,
            score_weights=score_weights,
            single_agent_tasks=single_agent_tasks,
        )
        if group:
            commit_group_assignment(assignment, group)
        elif use_value_replacement and not skip_congestion:
            _try_value_aware_replacement(
                task,
                agents,
                tasks,
                assignment,
                init_skip_congestion=init_skip_congestion,
                subtask_order=subtask_order,
                require_positive_revenue=require_positive_revenue,
                require_skill_match=require_skill_match,
                score_weights=score_weights,
                single_agent_tasks=single_agent_tasks,
            )

    if skip_congestion:
        rebuild_agent_state(agents, assignment)
        return assignment

    agent_map = {agent.id: agent for agent in agents}
    agents_to_check = set(agent_map)
    conflict_set = set()

    for _ in range(100):
        for agent_id in list(agents_to_check):
            if not _agent_is_safe(agent_id, agent_map[agent_id], assignment):
                conflict_set.add(agent_id)
        agents_to_check.clear()

        if not conflict_set:
            break

        for agent_id in list(conflict_set):
            agent = agent_map[agent_id]
            while len(assignment.get(agent_id, [])) > 1 and not _agent_is_safe(
                agent_id, agent, assignment
            ):
                records = list(assignment.get(agent_id, []))
                if random_drop:
                    dropped = random.choice(records)
                else:
                    drop_cost_index = _build_assignment_index(assignment)
                    dropped = min(
                        records,
                        key=lambda rec: _record_drop_cost(
                            rec,
                            agents,
                            assignment,
                            tabu,
                            score_weights=score_weights,
                            require_skill_match=require_skill_match,
                            assignment_index=drop_cost_index,
                        ),
                    )
                _, changed_agents = _replace_or_remove_record(
                    dropped,
                    agents,
                    assignment,
                    tabu,
                    skip_repick=skip_repick,
                    score_weights=score_weights,
                    require_skill_match=require_skill_match,
                )
                agents_to_check.update(changed_agents)
                agents_to_check.add(agent_id)
        conflict_set.clear()

    rebuild_agent_state(agents, assignment)
    return assignment


def run_CHASE_full(agents, tasks):
    return _run_chase_group(agents, tasks)


def run_CHASE_ablation(
    agents,
    tasks,
    skip_congestion=False,
    skip_repick=False,
    random_drop=False,
    init_skip_congestion=None,
    task_order="workload_asc",
    subtask_order="workload_desc",
    require_positive_revenue=True,
    require_skill_match=True,
    score_weights=None,
    use_value_replacement=False,
    single_agent_tasks=False,
):
    return _run_chase_group(
        agents,
        tasks,
        skip_congestion=skip_congestion,
        skip_repick=skip_repick,
        random_drop=random_drop,
        init_skip_congestion=init_skip_congestion,
        task_order=task_order,
        subtask_order=subtask_order,
        require_positive_revenue=require_positive_revenue,
        require_skill_match=require_skill_match,
        score_weights=score_weights,
        use_value_replacement=use_value_replacement,
        single_agent_tasks=single_agent_tasks,
    )


CHASE_ABLATION_SPECS = {
    "CHASE": {
        "label": "Full CHASE",
        "color": "#0868a6",
        "kwargs": {},
    },
    "CHASE_NO_CONG": {
        "label": "w/o Congestion Awareness",
        "color": "#439cc4",
        "kwargs": {"skip_congestion": True},
    },
    "CHASE_NO_COALITION": {
        "label": "w/o Collaborative Coalition",
        "color": "#7bc6be",
        "kwargs": {"single_agent_tasks": True},
    },
}

CHASE_CORE_ABLATION_MODES = [
    "CHASE",
    "CHASE_NO_CONG",
    "CHASE_NO_COALITION",
]


def _run_chase_mode(mode, agents, tasks):
    spec = CHASE_ABLATION_SPECS.get(mode)
    if spec is None:
        return False
    run_CHASE_ablation(agents, tasks, **spec["kwargs"])
    return True


BASELINE_BRTOA_ACTION_TO_TYPE = {
    0: "Low",
    1: "Mid",
    2: "High",
}

BASELINE_TIER_FALLBACK_PENALTY = 0.05


def _baseline_candidate_agents(
    agents,
    subtask,
    assignment,
    tentative=None,
    agent_type=None,
    preferred_agent_type=None,
    used_agent_ids=None,
    agent_loads=None,
    agent_counts=None,
):
    tentative = tentative or []
    used_agent_ids = used_agent_ids or set()
    candidates = []
    for agent in agents:
        if agent.id in used_agent_ids:
            continue
        if agent_type is not None and agent.type != agent_type:
            continue
        if not agent.can_execute(subtask.skill):
            continue
        if agent_loads is None or agent_counts is None:
            agent_records = list(assignment.get(agent.id, []))
            agent_records.extend(record for record in tentative if record.agent.id == agent.id)
            current_load = _agent_load_from_records(agent, agent_records)
            current_count = len(agent_records)
        else:
            current_load = agent_loads.get(agent.id, 0.0)
            current_count = agent_counts.get(agent.id, 0)
        omega = subtask.wk / agent.capacity
        if current_load + omega > 1.0:
            continue
        projected_count = current_count + 1
        projected_load = current_load + omega
        projected_time = calc_subtask_real_time(
            subtask,
            agent,
            projected_count,
            projected_load,
        )
        tier_penalty = (
            0
            if preferred_agent_type is None or agent.type == preferred_agent_type
            else 1
        )
        candidate_score = projected_time + tier_penalty * BASELINE_TIER_FALLBACK_PENALTY
        candidates.append((candidate_score, projected_time, tier_penalty, agent))
    return candidates


def _baseline_assign_task_min_delay(
    task,
    agents,
    assignment,
    *,
    agent_type=None,
    preferred_agent_type=None,
    preferred_agent=None,
    random_choice=False,
    prefer_tier_first=False,
    prefer_agent_first=False,
):
    tentative = []
    used_agent_ids = assigned_agent_ids_for_task(assignment, task)
    agent_loads, agent_counts = _assignment_agent_stats(assignment)
    for subtask in task.subtasks:
        candidates = _baseline_candidate_agents(
            agents,
            subtask,
            assignment,
            tentative=tentative,
            agent_type=agent_type,
            preferred_agent_type=preferred_agent_type,
            used_agent_ids=used_agent_ids,
            agent_loads=agent_loads,
            agent_counts=agent_counts,
        )
        if not candidates:
            return None
        if random_choice:
            *_, selected_agent = random.choice(candidates)
        elif prefer_agent_first and preferred_agent is not None:
            candidates.sort(
                key=lambda item: (
                    0 if item[3].id == preferred_agent.id else 1,
                    item[1],
                    item[3].id,
                )
            )
            selected_agent = candidates[0][3]
        elif prefer_tier_first and preferred_agent_type is not None:
            candidates.sort(key=lambda item: (item[2], item[1], item[3].id))
            selected_agent = candidates[0][3]
        else:
            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    0 if preferred_agent is not None and item[3].id == preferred_agent.id else 1,
                    item[3].id,
                )
            )
            selected_agent = candidates[0][3]
        record = SubtaskAssignment(task, subtask, selected_agent)
        tentative.append(record)
        used_agent_ids.add(selected_agent.id)
        agent_loads[selected_agent.id] = agent_loads.get(selected_agent.id, 0.0) + assignment_load(
            record,
            selected_agent,
        )
        agent_counts[selected_agent.id] = agent_counts.get(selected_agent.id, 0) + 1
    return tentative


def _assignment_signature(assignment):
    return tuple(
        sorted(
            (record.task.id, record.subtask.id, agent_id)
            for agent_id, records in assignment.items()
            for record in records
        )
    )


def _assignment_metrics(tasks, assignment):
    revenue = _assignment_total_revenue(tasks, assignment)
    success_count = sum(
        1
        for task in tasks
        if _task_revenue_from_assignment(task, assignment) > 0
    )
    return revenue, success_count / len(tasks) if tasks else 0.0


def _run_group_greedy(agents, tasks, *, random_choice=False, mct=False):
    assignment = _empty_assignment(agents)
    scheduled_tasks = list(tasks)
    if random_choice:
        random.shuffle(scheduled_tasks)
    else:
        scheduled_tasks.sort(key=lambda task: task.wk)

    for task in scheduled_tasks:
        group = _baseline_assign_task_min_delay(
            task,
            agents,
            assignment,
            random_choice=random_choice and not mct,
        )
        if group:
            commit_group_assignment(assignment, group)

    rebuild_agent_state(agents, assignment)
    return assignment


class Baseline_BRTOA:
    """BRTOA without DE, adapted to task-tier offloading actions."""

    ACTION_TO_TYPE = BASELINE_BRTOA_ACTION_TO_TYPE
    ACTIONS = tuple(ACTION_TO_TYPE)

    def __init__(self, agents, tasks):
        self.agents = agents
        self.tasks = tasks
        self.max_iterations = 3000
        self.initial_actions = {task.id: 0 for task in tasks}
        self.current_actions = dict(self.initial_actions)
        self.final_actions = dict(self.initial_actions)
        self.action_history = [dict(self.initial_actions)]
        self.iterations = 0

    def _assign_task_for_action(self, task, action, assignment):
        return _baseline_assign_task_min_delay(
            task,
            self.agents,
            assignment,
            preferred_agent_type=self.ACTION_TO_TYPE[action],
            prefer_tier_first=True,
        )

    def _build_assignment_for_actions(self, actions):
        assignment = _empty_assignment(self.agents)
        for task in sorted(self.tasks, key=lambda item: item.wk):
            group = self._assign_task_for_action(task, actions[task.id], assignment)
            if group:
                commit_group_assignment(assignment, group)
        return assignment

    def _action_metrics(
        self,
        task,
        action,
        base_assignment,
        current_assignment=None,
        removed_records=None,
        current_index=None,
    ):
        group = self._assign_task_for_action(task, action, base_assignment)
        if not group:
            return None

        projected = clone_assignment(base_assignment)
        commit_group_assignment(projected, group)
        records = task_records_from_assignment(projected, task)
        if not task_is_covered(task, records):
            return None
        delay = calc_task_completion_time(task, records, assignment=projected)
        if not np.isfinite(delay):
            return None
        revenue = calc_task_revenue(task, records, assignment=projected)
        delta = revenue
        if current_assignment is not None:
            delta = _replacement_delta(
                current_assignment,
                projected,
                removed_records or [],
                group,
                current_index=current_index,
            )
        return {
            "action": action,
            "delay": delay,
            "revenue": revenue,
            "delta": delta,
            "group": group,
        }

    def _action_delay(self, task, action, actions):
        assignment = self._build_assignment_for_actions(actions)
        base_assignment = clone_assignment(assignment)
        removed_records = remove_task_from_assignment(base_assignment, task.id)
        metrics = self._action_metrics(task, action, base_assignment)
        if metrics is None or metrics["delay"] > task.deadline:
            return float("inf")
        return metrics["delay"]

    def _feasible_actions(self, task, actions, assignment=None):
        if assignment is None:
            assignment = self._build_assignment_for_actions(actions)
        current_index = _build_assignment_index(assignment)
        base_assignment = clone_assignment(assignment)
        removed_records = remove_task_from_assignment(base_assignment, task.id)
        feasible = []
        for action in self.ACTIONS:
            metrics = self._action_metrics(
                task,
                action,
                base_assignment,
                current_assignment=assignment,
                removed_records=removed_records,
                current_index=current_index,
            )
            if metrics is not None:
                feasible.append(metrics)
        return feasible

    def _current_task_metrics(self, task, assignment, action):
        records = task_records_from_assignment(assignment, task)
        if not task_is_covered(task, records):
            return {
                "action": action,
                "delay": float("inf"),
                "revenue": 0.0,
                "delta": 0.0,
                "group": None,
            }
        delay = calc_task_completion_time(task, records, assignment=assignment)
        revenue = calc_task_revenue(task, records, assignment=assignment)
        return {
            "action": action,
            "delay": delay,
            "revenue": revenue,
            "delta": 0.0,
            "group": records,
        }

    def _response_key(self, metrics):
        delay = metrics["delay"]
        return (
            metrics.get("delta", metrics["revenue"]),
            metrics["revenue"],
            -delay if np.isfinite(delay) else -float("inf"),
            -metrics["action"],
        )

    def _actions_signature(self, actions):
        return tuple(sorted(actions.items()))

    def _best_response(self, task, actions, assignment=None):
        if assignment is None:
            assignment = self._build_assignment_for_actions(actions)
        current = self._current_task_metrics(task, assignment, actions[task.id])
        feasible = self._feasible_actions(task, actions, assignment=assignment)
        if not feasible:
            return current
        feasible.sort(key=self._response_key, reverse=True)
        best = feasible[0]
        if self._response_key(best) > self._response_key(current):
            return best
        return current

    def run(self):
        actions = dict(self.initial_actions)
        assignment = _empty_assignment(self.agents)

        for task in sorted(self.tasks, key=lambda item: item.wk):
            candidates = []
            for action in self.ACTIONS:
                metrics = self._action_metrics(task, action, assignment)
                if metrics is not None:
                    candidates.append(metrics)

            if not candidates:
                continue

            best = max(candidates, key=self._response_key)
            commit_group_assignment(assignment, best["group"])
            actions[task.id] = best["action"]
            self.action_history.append(dict(actions))

        self.iterations = 1
        self.current_actions = dict(actions)
        self.final_actions = dict(actions)
        return assignment


def _feasible_mask_for_subtask(task, subtask, agents, used_agent_ids):
    mask = np.zeros(len(agents), dtype=bool)
    for idx, agent in enumerate(agents):
        omega = subtask.wk / agent.capacity
        if (
            agent.id not in used_agent_ids
            and agent.can_execute(subtask.skill)
            and agent.current_load + omega <= 1.0
        ):
            mask[idx] = True
    return mask


def _run_fldrl_group(agents, tasks):
    fl_drl_model = get_fl_drl_model()
    for task in sorted(tasks, key=lambda t: t.wk):
        used_agent_ids = set()
        task_records = []
        failed = False
        for idx, subtask in enumerate(task.subtasks):
            progress = idx / max(1, len(task.subtasks))
            state = fl_drl_model.build_state(task, subtask, agents, progress=progress)
            feasible_mask = _feasible_mask_for_subtask(task, subtask, agents, used_agent_ids)
            selected_idx = fl_drl_model.choose_action(state, feasible_mask)
            if selected_idx is None:
                failed = True
                break
            target_agent = agents[selected_idx]
            if not feasible_mask[selected_idx]:
                failed = True
                break
            task_records.append(add_live_assignment(target_agent, task, subtask))
            used_agent_ids.add(target_agent.id)

        if failed:
            remove_live_records(agents, task_records)


class DyLAN:
    """DyLAN two-stage T-FFN surrogate for task scheduling."""

    def __init__(
        self,
        agents,
        tasks,
        T_max=5,
        K_select=None,
        K_keep=None,
        consistency_theta=2 / 3,
        t_reform=None,
        task_type="decision-making",
    ):
        self.agents = list(agents)
        self.tasks = list(tasks)
        self.T_max = T_max
        self.K_select = K_select if K_select is not None else max(1, len(agents) // 2)
        self.K_keep = K_keep if K_keep is not None else max(1, len(agents) // 2)
        self.theta = consistency_theta
        self.t_reform = set(t_reform if t_reform is not None else range(2, T_max + 1))
        self.task_type = task_type
        self.V_layers = []
        self.E_layers = []
        self.messages = {}
        self.peer_scores = {}
        self.importance_scores = {}
        self.preliminary = None
        self.final_output = None
        self.selected_agents = []
        self.selected_covers_required = False

    def _agent_response(self, active_agents, preferred_agent):
        assignment = _empty_assignment(self.agents)
        for task in sorted(self.tasks, key=lambda t: t.wk):
            group = _baseline_assign_task_min_delay(
                task,
                active_agents,
                assignment,
                preferred_agent=preferred_agent,
                prefer_agent_first=True,
            )
            if group:
                commit_group_assignment(assignment, group)

        revenue, success_rate = _assignment_metrics(self.tasks, assignment)
        return {
            "assignment": assignment,
            "revenue": revenue,
            "success_rate": success_rate,
            "signature": _assignment_signature(assignment),
        }

    def _llm_ranker(self, active_agents, layer_nodes, messages):
        scored = []
        for agent in active_agents:
            node = next((node for node in layer_nodes if node[1] == agent.id), None)
            revenue = messages.get(node, {}).get("revenue", 0.0) if node else 0.0
            scored.append((revenue, agent))
        scored.sort(key=lambda item: (item[0], item[1].id), reverse=True)
        return [agent for _, agent in scored]

    def _largest_consistency_group(self, responses):
        groups = defaultdict(list)
        for response in responses:
            groups[response["signature"]].append(response)
        if not groups:
            return []
        return max(groups.values(), key=len)

    def _early_stopping(self, responses):
        if not responses:
            return False, []
        largest_group = self._largest_consistency_group(responses)
        return (len(largest_group) / len(responses)) > self.theta, largest_group

    def _post_process(self, responses):
        if not responses:
            return {
                "assignment": _empty_assignment(self.agents),
                "revenue": 0.0,
                "success_rate": 0.0,
                "signature": tuple(),
            }
        largest_group = self._largest_consistency_group(responses)
        candidates = largest_group if largest_group else responses
        return max(candidates, key=lambda response: response["revenue"])

    def _normalized_peer_scores(self, predecessor_nodes, messages):
        if not predecessor_nodes:
            return {}
        raw_scores = {
            node: max(0.0, messages[node]["revenue"])
            for node in predecessor_nodes
        }
        total = sum(raw_scores.values())
        if total <= 0:
            uniform = 1.0 / len(predecessor_nodes)
            return {node: uniform for node in predecessor_nodes}
        return {node: score / total for node, score in raw_scores.items()}

    def _dylan_inference(self, agents, collect_peer_score=False):
        V_layers = []
        E_layers = []
        messages = {}
        peer_scores = {}
        active_agents = list(agents)

        first_layer = []
        for agent in active_agents:
            node = (1, agent.id)
            first_layer.append(node)
            messages[node] = self._agent_response(active_agents, agent)
        V_layers.append(first_layer)

        should_stop, stop_group = self._early_stopping(
            [messages[node] for node in first_layer]
        )
        if should_stop:
            output = self._post_process(stop_group)
            return (V_layers, E_layers), messages, peer_scores, output

        for t in range(2, self.T_max + 1):
            previous_agents = list(active_agents)
            previous_layer = V_layers[-1]
            if t in self.t_reform:
                ranked_agents = self._llm_ranker(previous_agents, previous_layer, messages)
                active_agents = ranked_agents[: self.K_keep]

            current_layer = []
            current_edges = set()
            for agent in active_agents:
                node = (t, agent.id)
                current_layer.append(node)
                for prev_node in previous_layer:
                    current_edges.add((prev_node, node))
            E_layers.append(current_edges)

            for node in current_layer:
                predecessor_nodes = [
                    prev_node
                    for prev_node, dst_node in current_edges
                    if dst_node == node
                ]
                if collect_peer_score:
                    scores = self._normalized_peer_scores(predecessor_nodes, messages)
                    for prev_node, score in scores.items():
                        peer_scores[(prev_node, node)] = score
                agent = next(agent for agent in active_agents if agent.id == node[1])
                messages[node] = self._agent_response(active_agents, agent)

            V_layers.append(current_layer)
            should_stop, stop_group = self._early_stopping(
                [messages[node] for node in current_layer]
            )
            if should_stop:
                output = self._post_process(stop_group)
                return (V_layers, E_layers), messages, peer_scores, output

        output = self._post_process([messages[node] for node in V_layers[-1]])
        return (V_layers, E_layers), messages, peer_scores, output

    def _compute_agent_importance_score(self, graph, messages, peer_scores, trial_output):
        V_layers, E_layers = graph
        contributions = {
            node: 0.0
            for layer in V_layers
            for node in layer
        }
        if not V_layers:
            return {agent.id: 0.0 for agent in self.agents}

        final_nodes = V_layers[-1]
        consistent_nodes = [
            node
            for node in final_nodes
            if messages[node]["signature"] == trial_output["signature"]
        ]
        if not consistent_nodes:
            consistent_nodes = list(final_nodes)
        if consistent_nodes:
            seed_value = 1.0 / len(consistent_nodes)
            for node in consistent_nodes:
                contributions[node] = seed_value

        for edges in reversed(E_layers):
            for prev_node, next_node in edges:
                contributions[prev_node] += contributions[next_node] * peer_scores.get(
                    (prev_node, next_node),
                    0.0,
                )

        importance = {agent.id: 0.0 for agent in self.agents}
        for node, contribution in contributions.items():
            importance[node[1]] = importance.get(node[1], 0.0) + contribution
        return importance

    def _required_skills(self):
        return {
            subtask.skill
            for task in self.tasks
            for subtask in task.subtasks
        }

    def _covered_skills(self, agents):
        covered = set()
        for agent in agents:
            covered.update(agent.skills)
        return covered

    def _agents_cover_required_skills(self, agents):
        return self._required_skills() <= self._covered_skills(agents)

    def _repair_selected_agents(self, selected_agents, k):
        selected = list(selected_agents[:k])
        selected_ids = {agent.id for agent in selected}
        if self._agents_cover_required_skills(selected):
            return selected, True

        candidates = [
            agent
            for agent in sorted(
                self.agents,
                key=lambda item: (self.importance_scores.get(item.id, 0.0), item.id),
                reverse=True,
            )
            if agent.id not in selected_ids
        ]

        while candidates and not self._agents_cover_required_skills(selected):
            missing_skills = self._required_skills() - self._covered_skills(selected)
            candidate = next(
                (
                    agent
                    for agent in candidates
                    if any(agent.can_execute(skill) for skill in missing_skills)
                ),
                None,
            )
            if candidate is None:
                break

            current_coverage = len(self._covered_skills(selected))
            replacement_idx = None
            for idx, victim in sorted(
                enumerate(selected),
                key=lambda item: (
                    self.importance_scores.get(item[1].id, 0.0),
                    item[1].id,
                ),
            ):
                trial = selected[:idx] + selected[idx + 1:] + [candidate]
                if self._agents_cover_required_skills(trial):
                    replacement_idx = idx
                    break
                if len(self._covered_skills(trial)) > current_coverage:
                    replacement_idx = idx
                    break

            if replacement_idx is None:
                replacement_idx = min(
                    range(len(selected)),
                    key=lambda idx: (
                        self.importance_scores.get(selected[idx].id, 0.0),
                        selected[idx].id,
                    ),
                )

            selected_ids.discard(selected[replacement_idx].id)
            selected[replacement_idx] = candidate
            selected_ids.add(candidate.id)
            candidates = [agent for agent in candidates if agent.id != candidate.id]

        return selected, self._agents_cover_required_skills(selected)

    def _output_key(self, output):
        return (output.get("revenue", 0.0), output.get("success_rate", 0.0))

    def run_team_optimization(self, k=None):
        if self.preliminary is None:
            graph, messages, peer_scores, output = self._dylan_inference(
                self.agents,
                collect_peer_score=True,
            )
            self.preliminary = (graph, messages, peer_scores, output)
            self.importance_scores = self._compute_agent_importance_score(
                graph,
                messages,
                peer_scores,
                output,
            )
        if k is None:
            k = self.K_select
        ranked = sorted(
            self.agents,
            key=lambda agent: (self.importance_scores.get(agent.id, 0.0), agent.id),
            reverse=True,
        )
        selected, covers_required = self._repair_selected_agents(ranked[:k], k)
        self.selected_agents = selected
        self.selected_covers_required = covers_required
        return selected, dict(self.importance_scores)

    def run_task_solving(self):
        selected_agents, _ = self.run_team_optimization(self.K_select)
        preliminary_graph, preliminary_messages, preliminary_peer_scores, preliminary_output = (
            self.preliminary
        )

        if self.selected_covers_required:
            graph, messages, peer_scores, output = self._dylan_inference(
                selected_agents,
                collect_peer_score=False,
            )
            if self._output_key(preliminary_output) > self._output_key(output):
                graph, messages, peer_scores, output = (
                    preliminary_graph,
                    preliminary_messages,
                    preliminary_peer_scores,
                    preliminary_output,
                )
        else:
            graph, messages, peer_scores, output = (
                preliminary_graph,
                preliminary_messages,
                preliminary_peer_scores,
                preliminary_output,
            )

        self.V_layers, self.E_layers = graph
        self.messages = messages
        self.peer_scores = peer_scores
        self.final_output = output
        rebuild_agent_state(self.agents, output["assignment"])
        return {agent.id: agent.concurrent_tasks for agent in self.agents}


def _collect_task_metrics(agents, tasks):
    total_revenue = 0.0
    success_count = 0
    for task in tasks:
        records = task_records_from_agents(agents, task)
        revenue = calc_task_revenue(task, records)
        total_revenue += revenue
        if revenue > 0:
            success_count += 1
    return total_revenue, success_count


def run_algorithm(mode, agents, tasks):
    for agent in agents:
        agent.reset()
    start_time = time.perf_counter()

    if mode == "IRS":
        _run_group_greedy(agents, tasks, random_choice=True)
    elif _run_chase_mode(mode, agents, tasks):
        pass
    elif mode == "FL_DRL":
        _run_fldrl_group(agents, tasks)
    elif mode == "BRTOA":
        assignment = Baseline_BRTOA(agents, tasks).run()
        rebuild_agent_state(agents, assignment)
    elif mode == "MCT":
        _run_group_greedy(agents, tasks, mct=True)
    elif mode == "DyLAN":
        DyLAN(agents, tasks).run_task_solving()
    else:
        raise ValueError(f"Unknown algorithm mode: {mode}")

    total_revenue, success_count = _collect_task_metrics(agents, tasks)
    end_time = (time.perf_counter() - start_time) * 1000

    utilization = {"High": [], "Mid": [], "Low": []}
    concurrency_counts = {"High": [], "Mid": [], "Low": []}
    for agent in agents:
        utilization.setdefault(agent.type, []).append(agent.current_load)
        concurrency_counts.setdefault(agent.type, []).append(len(agent.concurrent_tasks))

    avg_util = {key: np.mean(value) if value else 0 for key, value in utilization.items()}
    avg_conc = {key: np.mean(value) if value else 0 for key, value in concurrency_counts.items()}

    return {
        "revenue": total_revenue,
        "success_rate": success_count / len(tasks) if tasks else 0,
        "time": end_time,
        "utilization": avg_util,
        "concurrency": avg_conc,
    }


# Experiment runners

import multiprocessing
from functools import partial

def _worker_full_exp(n_task, trial_idx, target_snapshot_task, real_trace_data=None, real_agent_dist=None):

    random.seed()
    np.random.seed()

    agents = create_agents_from_real_data(AGENT_COUNT, real_distribution=real_agent_dist)
    tasks = create_tasks_from_real_data(n_task, real_data=real_trace_data)

    res_mct    = run_algorithm('MCT',    agents, tasks)
    res_CHASE  = run_algorithm('CHASE',  agents, tasks)
    res_irs    = run_algorithm('IRS',    agents, tasks)
    res_dylan  = run_algorithm('DyLAN',  agents, tasks)
    res_fldrl  = run_algorithm('FL_DRL', agents, tasks)
    res_brtoa  = run_algorithm('BRTOA',  agents, tasks)

    snapshot = None
    if n_task == target_snapshot_task and trial_idx == 0:
        snapshot = {
            'MCT':    res_mct['utilization'],
            'CHASE':  res_CHASE['utilization'],
            'IRS':    res_irs['utilization'],
            'DyLAN':  res_dylan['utilization'],
            'FL_DRL': res_fldrl['utilization'],
            'BRTOA':  res_brtoa['utilization'],
        }

    return {
        'MCT': res_mct, 'CHASE': res_CHASE, 'IRS': res_irs,
        'DyLAN': res_dylan, 'FL_DRL': res_fldrl, 'BRTOA': res_brtoa,
        'snapshot': snapshot
    }

def _worker_ablation(n_task, ablation_modes, trial_idx, real_trace_data=None, real_agent_dist=None):
    random.seed()
    np.random.seed()

    agents = create_agents_from_real_data(AGENT_COUNT, real_distribution=real_agent_dist)
    tasks = create_tasks_from_real_data(n_task, real_data=real_trace_data)

    res = {}
    for abl_mode in ablation_modes:
        res[abl_mode] = run_algorithm(abl_mode, agents, tasks)
    return res

def _worker_scale(n_agent, n_task, trial_idx, real_trace_data=None, real_agent_dist=None):
    random.seed()
    np.random.seed()

    agents = create_agents_from_real_data(n_agent, real_distribution=real_agent_dist)
    tasks = create_tasks_from_real_data(n_task, real_data=real_trace_data)

    return {
        'MCT':    run_algorithm('MCT',    agents, tasks),
        'CHASE':  run_algorithm('CHASE',  agents, tasks),
        'IRS':    run_algorithm('IRS',    agents, tasks),
        'DyLAN':  run_algorithm('DyLAN',  agents, tasks),
        'FL_DRL': run_algorithm('FL_DRL', agents, tasks),
        'BRTOA':  run_algorithm('BRTOA',  agents, tasks)
    }


def _run_algorithms_no_fldrl(base_agents, tasks, algorithms=None):
    algorithms = algorithms or HIGH_PRESSURE_ALGORITHMS
    results = {}
    for alg in algorithms:
        fresh_agents = [
            Agent(agent.id, agent.type, agent.capacity, agent.output)
            for agent in base_agents
        ]
        results[alg] = run_algorithm(alg, fresh_agents, tasks)
    return results


def _worker_high_pressure(n_task, trial_idx, real_trace_data=None, real_agent_dist=None,
                          deadline=HIGH_PRESSURE_DEADLINE,
                          workload_scale=HIGH_PRESSURE_WORKLOAD_SCALE):
    random.seed()
    np.random.seed()

    base_agents = create_agents_from_real_data(AGENT_COUNT, real_distribution=real_agent_dist)
    base_tasks = create_tasks_from_real_data(n_task, real_data=real_trace_data)
    tasks = clone_tasks_for_scenario(
        base_tasks,
        deadline=deadline,
        workload_scale=workload_scale,
    )
    return _run_algorithms_no_fldrl(base_agents, tasks)


def run_high_pressure_experiment(num_trials=HIGH_PRESSURE_TRIALS,
                                 task_nums=None,
                                 deadline=HIGH_PRESSURE_DEADLINE,
                                 workload_scale=HIGH_PRESSURE_WORKLOAD_SCALE):
    task_nums = task_nums or TASK_NUM_RANGE
    print(">>> Running high-pressure experiment (no FL-DRL)")
    print(f"    deadline={deadline}, workload_scale={workload_scale}, trials={num_trials}")

    real_trace_data = load_real_trace_data()
    real_agent_dist = load_agent_distribution()
    stats = {
        "scenario": {
            "deadline": deadline,
            "workload_scale": workload_scale,
            "trials": num_trials,
        },
        "tasks": list(task_nums),
        "algorithms": list(HIGH_PRESSURE_ALGORITHMS),
        "results": {},
    }

    for n_task in task_nums:
        buckets = {
            alg: {"revenue": [], "success_rate": [], "time": []}
            for alg in HIGH_PRESSURE_ALGORITHMS
        }
        for trial_idx in range(num_trials):
            random.seed(20260518 + trial_idx + n_task * 31)
            np.random.seed(20260518 + trial_idx + n_task * 31)
            base_agents = create_agents_from_real_data(AGENT_COUNT, real_distribution=real_agent_dist)
            base_tasks = create_tasks_from_real_data(n_task, real_data=real_trace_data)
            tasks = clone_tasks_for_scenario(
                base_tasks,
                deadline=deadline,
                workload_scale=workload_scale,
            )
            trial_results = _run_algorithms_no_fldrl(base_agents, tasks)
            for alg, result in trial_results.items():
                for metric in buckets[alg]:
                    buckets[alg][metric].append(result[metric])

        stats["results"][n_task] = {}
        for alg in HIGH_PRESSURE_ALGORITHMS:
            stats["results"][n_task][alg] = {}
            for metric, values in buckets[alg].items():
                arr = np.array(values)
                stats["results"][n_task][alg][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                }

        print(f"  Tasks={n_task}")
        for alg in HIGH_PRESSURE_ALGORITHMS:
            row = stats["results"][n_task][alg]
            print(
                f"    {alg:5s} revenue={row['revenue']['mean']:8.1f} "
                f"success={row['success_rate']['mean'] * 100:6.1f}% "
                f"time={row['time']['mean']:8.2f} ms"
            )

    return stats


def run_full_experiment():
    print(">>> 开始运行全量实验 (多进程并发加速版)...")


    num_workers = min(max(1, multiprocessing.cpu_count() - 2), 8)
    print(f"[*] 已检测到本地 CPU 核心数，将启用 {num_workers} 个并行进程进行计算。")

    results = {
        'tasks': TASK_NUM_RANGE,
        'MCT_rev': [], 'CHASE_rev': [], 'IRS_rev': [], 'DyLAN_rev': [], 'FL_DRL_rev': [], 'BRTOA_rev': [],
        'MCT_rev_std': [], 'CHASE_rev_std': [], 'IRS_rev_std': [], 'DyLAN_rev_std': [], 'FL_DRL_rev_std': [], 'BRTOA_rev_std': [],
        'MCT_succ': [], 'CHASE_succ': [], 'IRS_succ': [], 'DyLAN_succ': [], 'FL_DRL_succ': [], 'BRTOA_succ': [],
        'MCT_succ_std': [], 'CHASE_succ_std': [], 'IRS_succ_std': [], 'DyLAN_succ_std': [], 'FL_DRL_succ_std': [], 'BRTOA_succ_std': [],
        'MCT_time': [], 'CHASE_time': [], 'IRS_time': [], 'DyLAN_time': [], 'FL_DRL_time': [], 'BRTOA_time': [],
        'MCT_time_std': [], 'CHASE_time_std': [], 'IRS_time_std': [], 'DyLAN_time_std': [], 'FL_DRL_time_std': [], 'BRTOA_time_std': [],
        'snapshot_util_MCT': {'High':0, 'Mid':0, 'Low':0},
        'snapshot_util_CHASE': {'High':0, 'Mid':0, 'Low':0},
        'snapshot_util_IRS': {'High':0, 'Mid':0, 'Low':0},
        'snapshot_util_DyLAN': {'High':0, 'Mid':0, 'Low':0},
        'snapshot_util_FL_DRL': {'High':0, 'Mid':0, 'Low':0},
        'snapshot_util_BRTOA': {'High':0, 'Mid':0, 'Low':0},
        'snapshot_task_num': TASK_NUM_RANGE[-1]
    }





    stats = {'main': {}, 'ablation': {}, 'scale': {}}
    results['stats'] = stats

    target_snapshot_task = results['snapshot_task_num']


    global_trace_data = load_real_trace_data()
    global_agent_dist = load_agent_distribution()

    with multiprocessing.Pool(processes=num_workers) as pool:
        for idx, n_task in enumerate(TASK_NUM_RANGE):
            print(f"  [{idx+1}/{len(TASK_NUM_RANGE)}] Tasks={n_task}, 共{NUM_TRIALS}次试验并进行调度...")


            func = partial(_worker_full_exp, n_task, target_snapshot_task=target_snapshot_task,
                           real_trace_data=global_trace_data, real_agent_dist=global_agent_dist)
            trial_results = pool.map(func, range(NUM_TRIALS))


            temp_res = {'MCT': [], 'CHASE': [], 'IRS': [], 'DyLAN': [], 'FL_DRL': [], 'BRTOA': []}
            for r in trial_results:
                for alg in temp_res.keys():
                        temp_res[alg].append(r[alg])
                if r['snapshot'] is not None:
                    results['snapshot_util_MCT'] = r['snapshot']['MCT']
                    results['snapshot_util_CHASE'] = r['snapshot']['CHASE']
                    results['snapshot_util_IRS'] = r['snapshot']['IRS']
                    results['snapshot_util_DyLAN'] = r['snapshot']['DyLAN']
                    results['snapshot_util_FL_DRL'] = r['snapshot']['FL_DRL']
                    results['snapshot_util_BRTOA'] = r['snapshot']['BRTOA']


            stats['main'][n_task] = {}
            for metric in ['revenue', 'success_rate', 'time']:
                for alg in ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']:
                    vals_list = [r[metric] for r in temp_res[alg]]
                    arr = np.array(vals_list)
                    val = np.mean(arr)
                    std_val = np.std(arr)
                    if metric == 'revenue':
                        key = f'{alg}_rev'
                        results[key].append(val)
                        results[f'{alg}_rev_std'].append(std_val)

                        raw_key = f'{alg}_rev_raw'
                        if raw_key not in results:
                            results[raw_key] = []
                        results[raw_key].append(arr.tolist())
                    elif metric == 'success_rate':
                        key = f'{alg}_succ'
                        results[key].append(val)
                        results[f'{alg}_succ_std'].append(std_val)
                    else:
                        key = f'{alg}_time'
                        results[key].append(val)

                        results[f'{alg}_time_std'].append(std_val)

                    if alg not in stats['main'][n_task]:
                        stats['main'][n_task][alg] = {}
                    stats['main'][n_task][alg][metric] = {
                        'mean': float(np.mean(arr)),
                        'std':  float(std_val),
                        'min':  float(np.min(arr)),
                        'max':  float(np.max(arr)),
                        'median': float(np.median(arr)),
                    }


        ablation_modes = CHASE_CORE_ABLATION_MODES
        ablation_task_nums = [80, 100, 120]
        results['ablation'] = {
            'task_nums': ablation_task_nums,
            'revenue': {mode: [] for mode in ablation_modes},
            'success_rate': {mode: [] for mode in ablation_modes}
        }

        for n_task in ablation_task_nums:
            print(f"  Ablation: Tasks = {n_task}")
            func = partial(_worker_ablation, n_task, ablation_modes,
                           real_trace_data=global_trace_data, real_agent_dist=global_agent_dist)
            trial_results = pool.map(func, range(NUM_TRIALS))

            temp_rev = {mode: [] for mode in ablation_modes}
            temp_succ = {mode: [] for mode in ablation_modes}
            for r in trial_results:
                for mode in ablation_modes:
                    temp_rev[mode].append(r[mode]['revenue'])
                    temp_succ[mode].append(r[mode]['success_rate'])

            stats['ablation'][n_task] = {}
            for mode in ablation_modes:
                results['ablation']['revenue'][mode].append(np.mean(temp_rev[mode]))
                results['ablation']['success_rate'][mode].append(np.mean(temp_succ[mode]))
                rev_arr = np.array(temp_rev[mode])
                succ_arr = np.array(temp_succ[mode])
                stats['ablation'][n_task][mode] = {
                    'revenue': {'mean': float(np.mean(rev_arr)), 'std': float(np.std(rev_arr)),
                                'min': float(np.min(rev_arr)), 'max': float(np.max(rev_arr)),
                                'median': float(np.median(rev_arr))},
                    'success_rate': {'mean': float(np.mean(succ_arr)), 'std': float(np.std(succ_arr)),
                                     'min': float(np.min(succ_arr)), 'max': float(np.max(succ_arr)),
                                     'median': float(np.median(succ_arr))},
                }


        print(">>> 正在运行智能体规模扩展实验 (并发加速版)...")
        scale_agent_counts = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        scale_task_ratio = 3
        results['scale_agents'] = {
            'agent_counts': scale_agent_counts,
            'task_counts': [a * scale_task_ratio for a in scale_agent_counts],
            'revenue': {alg: [] for alg in ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']},
            'success_rate': {alg: [] for alg in ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']},
            'time': {alg: [] for alg in ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']},
        }

        for n_agent in scale_agent_counts:
            n_task = n_agent * scale_task_ratio
            print(f"  Scale: Agents={n_agent}, Tasks={n_task}")
            func = partial(_worker_scale, n_agent, n_task,
                           real_trace_data=global_trace_data, real_agent_dist=global_agent_dist)
            trial_results = pool.map(func, range(NUM_TRIALS))

            temp = {alg: {'rev':[], 'succ':[], 'time':[]} for alg in ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']}
            for r in trial_results:
                for alg in temp.keys():
                    temp[alg]['rev'].append(r[alg]['revenue'])
                    temp[alg]['succ'].append(r[alg]['success_rate'])
                    temp[alg]['time'].append(r[alg]['time'])

            stats['scale'][n_agent] = {}
            for alg in temp.keys():
                results['scale_agents']['revenue'][alg].append(np.mean(temp[alg]['rev']))
                results['scale_agents']['success_rate'][alg].append(np.mean(temp[alg]['succ']))
                results['scale_agents']['time'][alg].append(np.mean(temp[alg]['time']))
                stats['scale'][n_agent][alg] = {}
                for metric_name, metric_key in [('revenue','rev'),('success_rate','succ'),('time','time')]:
                    arr = np.array(temp[alg][metric_key])
                    stats['scale'][n_agent][alg][metric_name] = {
                        'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
                        'min': float(np.min(arr)), 'max': float(np.max(arr)),
                        'median': float(np.median(arr)),
                    }


    return results


def compute_significance_wilcoxon(res, algs=None, alpha=0.05):
























    from scipy.stats import wilcoxon

    if algs is None:
        algs = ['MCT', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']

    task_list = res['tasks']
    n_comparisons = len(algs)
    adj_alpha = alpha / n_comparisons

    sig_table = {}

    for i, n_task in enumerate(task_list):
        sig_table[n_task] = {}
        chase_samples = np.array(res.get('CHASE_rev_raw', [[]])[i])

        for baseline in algs:
            raw_key = f'{baseline}_rev_raw'
            baseline_samples = np.array(res.get(raw_key, [[]])[i])

            if len(chase_samples) < 10 or len(baseline_samples) < 10:

                sig_table[n_task][baseline] = {'p': float('nan'), 'star': 'n/a'}
                continue


            diff = chase_samples - baseline_samples


            if np.all(diff == 0):
                sig_table[n_task][baseline] = {'p': 1.0, 'star': 'ns'}
                continue


            try:
                stat, p_val = wilcoxon(diff, alternative='greater', zero_method='zsplit')
            except ValueError:
                p_val = 1.0


            if p_val < adj_alpha / 10:
                star = '***'
            elif p_val < adj_alpha / 2:
                star = '**'
            elif p_val < adj_alpha:
                star = '*'
            else:
                star = 'ns'

            sig_table[n_task][baseline] = {'p': float(p_val), 'star': star}

    return sig_table, adj_alpha


def plot_revenue_with_significance(res, output_dir, figsize_single,
                                   COLORS, MARKERS, STYLES, LABELS):










    sig_table, adj_alpha = compute_significance_wilcoxon(res)
    baselines = ['MCT', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']
    task_list = res['tasks']

    fig, ax = plt.subplots(figsize=figsize_single)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    alg_order = ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']
    y_arrays = {}

    for alg_key in alg_order:
        y_data = np.array(res[f'{alg_key}_rev'])
        y_arrays[alg_key] = y_data

        lw, ms, zorder = (3.5, 10, 10) if alg_key == 'CHASE' else (2.0, 7, 5)
        ls = STYLES[alg_key]
        if isinstance(ls, str) and ls.startswith('('):
            ls = eval(ls)

        ax.plot(task_list, y_data,
                marker=MARKERS[alg_key],
                linestyle=ls,
                color=COLORS[alg_key],
                label=LABELS[alg_key],
                linewidth=lw,
                markersize=ms,
                markeredgecolor='white',
                markeredgewidth=0.8,
                zorder=zorder)


    chase_y = y_arrays['CHASE']
    y_min, y_max = ax.get_ylim()

    all_y = np.concatenate([v for v in y_arrays.values()])
    y_span = all_y.max() - all_y.min()
    offset = y_span * 0.04

    star_level_order = {'***': 3, '**': 2, '*': 1, 'ns': 0, 'n/a': 0}

    for i, n_task in enumerate(task_list):
        cell = sig_table.get(n_task, {})

        worst_level = 3
        all_stars = []
        for bl in baselines:
            star = cell.get(bl, {}).get('star', 'ns')
            all_stars.append(star)
            worst_level = min(worst_level, star_level_order.get(star, 0))


        level_to_star = {3: '***', 2: '**', 1: '*', 0: None}
        display_star = level_to_star[worst_level]

        if display_star is not None:
            ax.annotate(
                display_star,
                xy=(task_list[i], chase_y[i] + offset),
                ha='center', va='bottom',
                fontsize=11, color='#C62828', fontweight='bold',
                annotation_clip=False
            )


    ax.set_xlabel('Number of Tasks', fontsize=13)
    ax.set_ylabel('Total Revenue', fontsize=13)
    ax.set_title('Total Revenue (★ = CHASE sig. better, Wilcoxon, Bonferroni)', fontsize=12, pad=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc',
              fontsize=10, loc='best')
    ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color='#999999')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)


    cur_ymin, cur_ymax = ax.get_ylim()
    ax.set_ylim(cur_ymin, cur_ymax + y_span * 0.12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig_5_1_Revenue_Sig.png'))
    plt.close()
    print('[OK] Fig_5_1_Revenue_Sig.png saved with Wilcoxon significance annotations.')


def plot_results(res):
    print(">>> Generating Publication-Quality Figures (Original 6)...")

    output_dir = str(FIGURES_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Plotting
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
        'CHASE':  '#E53935',
        'MCT':    '#FB8C00',
        'IRS':    '#26A69A',
        'DyLAN':  '#78909C',
        'FL_DRL': '#7E57C2',
        'BRTOA':  '#00897B',
    }
    MARKERS = {'CHASE': 'o', 'MCT': 's', 'IRS': '^', 'DyLAN': 'x', 'FL_DRL': 'd', 'BRTOA': 'p'}

    STYLES = {
        'CHASE':  '-',
        'MCT':    '--',
        'IRS':    '-.',
        'DyLAN':  ':',
        'FL_DRL': '(0,(3,1,1,1))',
        'BRTOA':  (0,(5,2)),
    }
    LABELS = {'CHASE': 'CHASE (Proposed)', 'MCT': 'MCT', 'IRS': 'IRS (Random)', 'DyLAN': 'DyLAN', 'FL_DRL': 'FL-DRL', 'BRTOA': 'BRTOA'}


    BAR_COLORS = {
        'MCT':    '#F58B58',
        'CHASE':  '#F2C458',
        'IRS':    '#5DD3C4',
        'DyLAN':  '#4A8F79',
        'FL_DRL': '#B68BC8',
        'BRTOA':  '#D2691E',
    }

    figsize_single = (7, 5)




    def _plot_line(x_data, y_keys, ylabel, clean_name, title=None, y_mult=1.0, yerr_keys=None):




        fig, ax = plt.subplots(figsize=figsize_single)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')


        err_map = {}
        if yerr_keys is not None:
            for alg_key, std_key in yerr_keys:
                err_map[alg_key] = std_key

        for alg_key, res_key in y_keys:
            y_data = np.array(res[res_key]) * y_mult


            if alg_key == 'CHASE':
                lw, ms, zorder = 3.5, 10, 10
            else:
                lw, ms, zorder = 2.0, 7, 5

            ls = STYLES[alg_key]

            if isinstance(ls, str) and ls.startswith('('):
                ls = eval(ls)

            if alg_key in err_map:

                yerr = np.array(res[err_map[alg_key]]) * y_mult
                ax.errorbar(x_data, y_data,
                            yerr=yerr,
                            marker=MARKERS[alg_key],
                            linestyle=ls,
                            color=COLORS[alg_key],
                            label=LABELS[alg_key],
                            linewidth=lw,
                            markersize=ms,
                            markeredgecolor='white',
                            markeredgewidth=0.8,
                            zorder=zorder,
                            capsize=4,
                            elinewidth=1.2,
                            capthick=1.2,
                            ecolor=COLORS[alg_key],
                            alpha=0.85)
            else:
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
        ax.set_ylabel(ylabel, fontsize=13)
        if title: ax.set_title(title, fontsize=15, pad=10)


        ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc',
                  fontsize=10, loc='best')


        ax.grid(True, linestyle='-', linewidth=0.4, alpha=0.25, color='#999999')
        ax.set_axisbelow(True)


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, clean_name + ".png"))
        plt.close()





    _plot_line(res['tasks'],
               [('MCT', 'MCT_rev'), ('CHASE', 'CHASE_rev'), ('IRS', 'IRS_rev'), ('DyLAN', 'DyLAN_rev'), ('FL_DRL', 'FL_DRL_rev'), ('BRTOA', 'BRTOA_rev')],
               'Total Revenue', 'Fig_5_1_Revenue', 'Total Revenue')

    plot_revenue_with_significance(res, output_dir, figsize_single,
                                   COLORS, MARKERS, STYLES, LABELS)




    _plot_line(res['tasks'],
               [('MCT', 'MCT_succ'), ('CHASE', 'CHASE_succ'), ('IRS', 'IRS_succ'), ('DyLAN', 'DyLAN_succ'), ('FL_DRL', 'FL_DRL_succ'), ('BRTOA', 'BRTOA_succ')],
               'Success Rate (%)', 'Fig_5_2_SuccessRate', 'Task Success Rate (%)', y_mult=100,
               yerr_keys=[('MCT', 'MCT_succ_std'), ('CHASE', 'CHASE_succ_std'), ('IRS', 'IRS_succ_std'),
                           ('DyLAN', 'DyLAN_succ_std'), ('FL_DRL', 'FL_DRL_succ_std'), ('BRTOA', 'BRTOA_succ_std')])





    task_target = 40
    if task_target in res['tasks']:
        idx = res['tasks'].index(task_target)
    else:
        idx = -1

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    algs = ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']
    time_vals = [res[f'{alg}_time'][idx] for alg in algs]
    time_stds = [res[f'{alg}_time_std'][idx] for alg in algs]
    x_pos = np.arange(len(algs))


    fig3_colors = ['#eaf3e2', '#b4deb6', '#7bc6be', '#439cc4', '#0868a6', '#ebd7b9']

    bar_labels = ['MCT', 'CHASE\n(Proposed)', 'IRS\n(Random)', 'DyLAN', 'FL-DRL', 'BRTOA']


    bar_width = 0.6
    for i, (alg, val, std, color) in enumerate(zip(algs, time_vals, time_stds, fig3_colors)):



        lower_err = min(std, val * 0.8)
        upper_err = std
        asym_err = [[lower_err], [upper_err]]

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


    for i, (val, std) in enumerate(zip(time_vals, time_stds)):
        top = val + std

        if val < 1:
            label = f'{val:.2f}'
        elif val < 100:
            label = f'{val:.1f}'
        else:
            label = f'{val:.0f}'
        ax.annotate(label,
                    xy=(x_pos[i], top),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='#333333')

    ax.set_ylabel('Time (ms, log scale)', fontsize=14)
    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_title('Algorithmic Time Cost', fontsize=16, pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=11)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)


    ax.grid(axis='y', linestyle='-', alpha=0.15, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig_5_3_TimeCost.png'))
    plt.close()




    snap_num = res['snapshot_task_num']
    labels = ['High', 'Mid', 'Low']
    x = np.arange(len(labels))
    w = 0.15


    def _plot_bar(res_prefix, ylabel, title, clean_name, y_mult=1.0):
        fig, ax = plt.subplots(figsize=(9, 6))

        algs = [('MCT', -2.5), ('CHASE', -1.5), ('IRS', -0.5), ('DyLAN', 0.5), ('FL_DRL', 1.5), ('BRTOA', 2.5)]

        for alg, offset_mult in algs:
            data_dict = res[f'{res_prefix}_{alg}']
            vals = [data_dict[k] * y_mult for k in labels]

            ax.bar(x + offset_mult*w, vals, w,
                   label=LABELS[alg], color=BAR_COLORS[alg],
                   edgecolor='black', linewidth=0.3)

        ax.set_title(f'{title} (Tasks={snap_num})', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlabel('Agent Capability Type', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        ax.legend(fontsize=12, frameon=True, loc='upper right')


        ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, clean_name + ".png"))
        plt.close()


    _plot_bar('snapshot_util', 'Utilization (%)', 'Avg Resource Utilization', 'Fig_5_4_Utilization', y_mult=100)



    ablation_data = res['ablation']
    task_nums = ablation_data['task_nums']
    x = np.arange(len(task_nums))
    ablation_modes = list(ablation_data['revenue'].keys())
    w_abl = min(0.18, 0.72 / max(1, len(ablation_modes)))
    fallback_colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(ablation_modes))))
    ablation_colors = {
        mode: CHASE_ABLATION_SPECS.get(mode, {}).get("color", fallback_colors[idx])
        for idx, mode in enumerate(ablation_modes)
    }
    ablation_labels = {
        mode: CHASE_ABLATION_SPECS.get(mode, {}).get("label", mode)
        for mode in ablation_modes
    }

    fig_width = max(9, 0.9 * len(ablation_modes) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    offsets = {
        mode: idx - (len(ablation_modes) - 1) / 2
        for idx, mode in enumerate(ablation_modes)
    }

    for mode, offset in offsets.items():
        vals = ablation_data['revenue'][mode]
        ax.bar(x + offset * w_abl, vals, w_abl,
               label=ablation_labels[mode], color=ablation_colors[mode],
               edgecolor='black', linewidth=0.3)

    ax.set_ylabel('Total Revenue', fontsize=14)
    ax.set_xlabel('Number of Tasks', fontsize=14)
    ax.set_title('Ablation Study: Revenue Impact', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in task_nums], fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, frameon=True, loc='upper left', ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig_5_7_Ablation_Revenue.png'))
    plt.close()


    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for mode, offset in offsets.items():
        vals = [v * 100 for v in ablation_data['success_rate'][mode]]
        ax.bar(x + offset * w_abl, vals, w_abl,
               label=ablation_labels[mode], color=ablation_colors[mode],
               edgecolor='black', linewidth=0.3)

    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_xlabel('Number of Tasks', fontsize=14)
    ax.set_title('Ablation Study: Success Rate Impact', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in task_nums], fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, frameon=True, loc='upper right', ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig_5_8_Ablation_SuccRate.png'))
    plt.close()




    print(">>> Generating Scale of Agents Figure...")

    scale_data = res['scale_agents']
    agent_counts = scale_data['agent_counts']








    x_arr = np.array(agent_counts, dtype=float)
    y_arr = np.array(scale_data['time']['CHASE'], dtype=float)


    log_x = np.log(x_arr)
    log_y = np.log(y_arr)
    k_fit, log_a_fit = np.polyfit(log_x, log_y, 1)
    y_fit = np.exp(log_a_fit) * x_arr ** k_fit


    anchor = y_arr[0]
    x0     = x_arr[0]
    ref_n1 = anchor * (x_arr / x0)
    ref_n2 = anchor * (x_arr / x0) ** 2


    fig, ax = plt.subplots(figsize=(7, 5.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


    ax.plot(x_arr, ref_n1,
            linestyle=':', linewidth=1.8, color='#9E9E9E',
            label=r'Reference $O(n)$', zorder=2)


    ax.plot(x_arr, ref_n2,
            linestyle='--', linewidth=1.4, color='#BDBDBD',
            label=r'Reference $O(n^2)$', zorder=2)


    ax.plot(x_arr, y_fit,
            linestyle=(0, (4, 2)), linewidth=1.6, color='#FB8C00',
            label=fr'Power-law fit $O(n^{{{k_fit:.2f}}})$', zorder=3)


    ax.plot(x_arr, y_arr,
            marker=MARKERS['CHASE'],
            linestyle=STYLES['CHASE'],
            color=COLORS['CHASE'],
            linewidth=3.0, markersize=9,
            markeredgecolor='white', markeredgewidth=0.8,
            label=LABELS['CHASE'] + ' (Actual)',
            zorder=5)


    ax.set_xscale('log')
    ax.set_yscale('log')


    ax.annotate(
        fr'Fitted exponent $k={k_fit:.2f}$',
        xy=(x_arr[-2], y_fit[-2]),
        xytext=(-80, 18), textcoords='offset points',
        fontsize=10, color='#E65100',
        arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', fc='#FFF8E1', ec='#FB8C00', lw=0.8)
    )


    ax.set_xlabel('Number of Agents (I)', fontsize=13)
    ax.set_ylabel('Time Cost (ms, log scale)', fontsize=13)
    ax.set_title('CHASE Scalability vs. Theoretical Complexity\n(Task:Agent = 3:1)',
                 fontsize=13, pad=10)

    ax.set_xticks(x_arr)
    ax.set_xticklabels([str(int(v)) for v in x_arr], fontsize=10)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc',
              fontsize=10, loc='upper left')

    ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.3, color='#999999')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.2, color='#cccccc')
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig_5_9_ScaleAgents_Time.png'))
    plt.close()

    print(f">>> All publication figures saved to: {output_dir}")



# Statistics export

def export_statistics_tables(res):

    output_dir = str(TABLES_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    stats = res.get('stats', {})
    algs = ['MCT', 'CHASE', 'IRS', 'DyLAN', 'FL_DRL', 'BRTOA']
    alg_display = {'MCT': 'MCT', 'CHASE': 'CHASE', 'IRS': 'IRS', 'DyLAN': 'DyLAN', 'FL_DRL': 'FL-DRL', 'BRTOA': 'BRTOA'}
    stat_keys = ['mean', 'std', 'min', 'max', 'median']




    for metric, metric_label, multiplier in [
        ('revenue', 'Revenue', 1.0),
        ('success_rate', 'SuccessRate(%)', 100.0),
        ('time', 'TimeCost(ms)', 1.0),
    ]:
        rows = []
        for n_task in sorted(stats.get('main', {}).keys()):
            for alg in algs:
                s = stats['main'][n_task].get(alg, {}).get(metric, {})
                if not s:
                    continue
                rows.append({
                    'Tasks': n_task,
                    'Algorithm': alg_display[alg],
                    'Mean': round(s['mean'] * multiplier, 4),
                    'Std': round(s['std'] * multiplier, 4),
                    'Min': round(s['min'] * multiplier, 4),
                    'Max': round(s['max'] * multiplier, 4),
                    'Median': round(s['median'] * multiplier, 4),
                })
        if rows:
            df = pd.DataFrame(rows)
            fname = os.path.join(output_dir, f'Table_Main_{metric_label}.csv')
            df.to_csv(fname, index=False, encoding='utf-8-sig')
            print(f"  [Table] {fname}  ({len(rows)} rows)")

    rows = []
    abl_modes_display = {
        mode: spec["label"]
        for mode, spec in CHASE_ABLATION_SPECS.items()
    }
    for n_task in sorted(stats.get('ablation', {}).keys()):
        for mode in stats['ablation'][n_task].keys():
            mode_stats = stats['ablation'][n_task].get(mode, {})
            for metric in ['revenue', 'success_rate']:
                s = mode_stats.get(metric, {})
                if not s:
                    continue
                mult = 100.0 if metric == 'success_rate' else 1.0
                rows.append({
                    'Tasks': n_task,
                    'Variant': abl_modes_display.get(mode, mode),
                    'Metric': metric,
                    'Mean': round(s['mean'] * mult, 4),
                    'Std': round(s['std'] * mult, 4),
                    'Min': round(s['min'] * mult, 4),
                    'Max': round(s['max'] * mult, 4),
                    'Median': round(s['median'] * mult, 4),
                })
    if rows:
        df = pd.DataFrame(rows)
        fname = os.path.join(output_dir, 'Table_Ablation.csv')
        df.to_csv(fname, index=False, encoding='utf-8-sig')
        print(f"  [Table] {fname}  ({len(rows)} rows)")


    rows = []
    for n_agent in sorted(stats.get('scale', {}).keys()):
        for alg in algs:
            alg_stats = stats['scale'][n_agent].get(alg, {})
            for metric in ['revenue', 'success_rate', 'time']:
                s = alg_stats.get(metric, {})
                if not s:
                    continue
                mult = 100.0 if metric == 'success_rate' else 1.0
                rows.append({
                    'Agents': n_agent,
                    'Tasks': n_agent * 3,
                    'Algorithm': alg_display[alg],
                    'Metric': metric,
                    'Mean': round(s['mean'] * mult, 4),
                    'Std': round(s['std'] * mult, 4),
                    'Min': round(s['min'] * mult, 4),
                    'Max': round(s['max'] * mult, 4),
                    'Median': round(s['median'] * mult, 4),
                })
    if rows:
        df = pd.DataFrame(rows)
        fname = os.path.join(output_dir, 'Table_ScaleAgents.csv')
        df.to_csv(fname, index=False, encoding='utf-8-sig')
        print(f"  [Table] {fname}  ({len(rows)} rows)")

    print(f">>> 所有统计表格已导出至: {output_dir}/")


def run_quick_smoke():
    """Run a tiny scheduling smoke test without requiring the full dataset or model."""
    real_data = {
        "workloads": [30, 40, 50, 60, 70],
        "cpis": [2.0, 2.0, 2.0, 2.0, 2.0],
    }
    agents = create_agents_from_real_data(5, real_distribution=None)
    tasks = create_tasks_from_real_data(5, real_data=real_data)
    algs = ["MCT", "CHASE", "IRS", "DyLAN", "BRTOA"]
    for alg in algs:
        fresh_agents = [Agent(a.id, a.type, a.capacity, a.output) for a in agents]
        res = run_algorithm(alg, fresh_agents, tasks)
        print(f"{alg}: revenue={res['revenue']:.3f}, success_rate={res['success_rate']:.3f}")


if __name__ == '__main__':


    multiprocessing.freeze_support()

    from project.cli import main as cli_main

    cli_main()
