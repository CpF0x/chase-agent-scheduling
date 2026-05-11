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

from project.config import BORG_TRACE_PATH, FIGURES_DIR, FL_DRL_MODEL_PATH as DEFAULT_MODEL_PATH, TABLES_DIR


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

class Baseline_BRTOA:
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


# Synthetic Alibaba-like workload






ALIBABA_TRACE_CACHE = None

def load_alibaba_synthetic_data(n_samples=50000, seed=42):






    global ALIBABA_TRACE_CACHE
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

class Agent:
    def __init__(self, uid, atype, cap, output):
        self.id = uid
        self.type = atype
        self.capacity = cap
        self.output = output
        self.current_load = 0.0
        self.concurrent_tasks = []

    def reset(self):
        self.current_load = 0.0
        self.concurrent_tasks = []

class Task:
    """Task model used by the experiments."""









    DEFAULT_CPI = 2.0

    def __init__(self, uid, real_workload=None, real_cpi=None):
        self.id = uid

        if real_workload is not None:
            self.wk = real_workload
        else:
            self.wk = random.uniform(*WORKLOAD_RANGE)


        if real_cpi is not None:
            self.cpi = real_cpi
        else:
            self.cpi = Task.DEFAULT_CPI

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


def calc_dynamic_output(agent, task):












    eta = calc_cpi_efficiency(task.cpi)
    return agent.capacity * eta



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

def calc_real_time(task, agent, concurrency, omega):














    base_t = task.wk / calc_dynamic_output(agent, task)
    alpha = get_alpha(agent)
    degradation = calc_degradation(omega, concurrency, alpha)
    return base_t * (1 + degradation)


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

def calc_revenue(task, agent, concurrency, omega=None):

























    import math
    if omega is None:
        omega = agent.current_load

    real_t = calc_real_time(task, agent, concurrency, omega)


    effective_deadline = task.deadline - task.tau_k


    x = REVENUE_DELTA * (real_t - effective_deadline)
    sig = 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


    utility = ((1.0 - sig) ** REVENUE_GAMMA) * task.base_val - REVENUE_COST
    return utility

def calc_utility(task, agent, concurrency, omega=None):

    return calc_revenue(task, agent, concurrency, omega)


# Task assignment strategy


def tas_find_agent(task, agents, assignment, skip_congestion=False):








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


# DyLAN baseline


class DyLAN:
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

def run_CHASE_full(agents, tasks):












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

def run_CHASE_ablation(agents, tasks, skip_congestion=False, skip_repick=False, random_drop=False):













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

def run_algorithm(mode, agents, tasks):
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


        ablation_modes = ['CHASE', 'CHASE_NO_CONG', 'CHASE_NO_REPICK', 'CHASE_NO_SMART']
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
    w_abl = 0.18
    ABLATION_COLORS = {
        'CHASE': '#0868a6',
        'CHASE_NO_CONG': '#439cc4',
        'CHASE_NO_REPICK': '#7bc6be',
        'CHASE_NO_SMART': '#b4deb6',
    }
    ABLATION_LABELS = {
        'CHASE': 'Full CHASE',
        'CHASE_NO_CONG': 'w/o Congestion',
        'CHASE_NO_REPICK': 'w/o Re-Pick',
        'CHASE_NO_SMART': 'w/o Smart-Drop',
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    offsets = {'CHASE': -1.5, 'CHASE_NO_CONG': -0.5, 'CHASE_NO_REPICK': 0.5, 'CHASE_NO_SMART': 1.5}

    for mode, offset in offsets.items():
        vals = ablation_data['revenue'][mode]
        ax.bar(x + offset * w_abl, vals, w_abl,
               label=ABLATION_LABELS[mode], color=ABLATION_COLORS[mode],
               edgecolor='black', linewidth=0.3)

    ax.set_ylabel('Total Revenue', fontsize=14)
    ax.set_xlabel('Number of Tasks', fontsize=14)
    ax.set_title('Ablation Study: Revenue Impact', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in task_nums], fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11, frameon=True, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Fig_5_7_Ablation_Revenue.png'))
    plt.close()


    fig, ax = plt.subplots(figsize=(9, 6))

    for mode, offset in offsets.items():
        vals = [v * 100 for v in ablation_data['success_rate'][mode]]
        ax.bar(x + offset * w_abl, vals, w_abl,
               label=ABLATION_LABELS[mode], color=ABLATION_COLORS[mode],
               edgecolor='black', linewidth=0.3)

    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_xlabel('Number of Tasks', fontsize=14)
    ax.set_title('Ablation Study: Success Rate Impact', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in task_nums], fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11, frameon=True, loc='upper right')
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
    ref_n2 = anchor * (x_arr / x0) ** 2
    ref_n3 = anchor * (x_arr / x0) ** 3


    fig, ax = plt.subplots(figsize=(7, 5.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


    ax.plot(x_arr, ref_n2,
            linestyle='--', linewidth=1.4, color='#BDBDBD',
            label=r'Reference $O(n^2)$', zorder=2)


    ax.plot(x_arr, ref_n3,
            linestyle=':', linewidth=1.8, color='#9E9E9E',
            label=r'Reference $O(n^3)$', zorder=2)


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
        fr'Fitted exponent $k={k_fit:.2f}$ (< 3)',
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

    abl_modes_display = {
        'CHASE': 'Full CHASE', 'CHASE_NO_CONG': 'w/o Congestion',
        'CHASE_NO_REPICK': 'w/o Re-Pick', 'CHASE_NO_SMART': 'w/o Smart-Drop',
    }
    for n_task in sorted(stats.get('ablation', {}).keys()):
        for mode in ['CHASE', 'CHASE_NO_CONG', 'CHASE_NO_REPICK', 'CHASE_NO_SMART']:
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
