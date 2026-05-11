#!/usr/bin/env python3
"""Train the FL-DRL scheduler."""














import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import copy
import argparse
from collections import deque


# Global training config

AGENT_COUNT = 40
DEADLINE = 0.9
WORKLOAD_RANGE = (25, 80)

# Revenue parameters
REVENUE_DELTA = 50.0
REVENUE_GAMMA = 3.0
REVENUE_COST  = 5.0


AGENT_DISTRIBUTION = {
    'High': {'ratio': 0.44, 'cap_range': (90, 120)},
    'Mid':  {'ratio': 0.26, 'cap_range': (50, 89)},
    'Low':  {'ratio': 0.30, 'cap_range': (25, 49)},
}


CPI_MODEL_PARAMS = {
    'eta_max': 0.75,
    'eta_min': 0.25,
    'cpi_ref': 2.0,
    'k': 1.2,
}


# Environment components

class SimpleAgent:
    """Lightweight training agent."""
    def __init__(self, uid, atype, cap):
        self.id = uid
        self.type = atype
        self.capacity = cap
        self.current_load = 0.0
        self.concurrent_tasks = []

    def reset(self):
        self.current_load = 0.0
        self.concurrent_tasks = []


def calc_tau(base_val, cost, delta, gamma):

    import math
    ratio = cost / base_val
    inner = 1.0 - ratio ** (1.0 / gamma)
    inner = max(1e-9, min(1.0 - 1e-9, inner))
    return (1.0 / delta) * math.log(inner / (1.0 - inner))

class SimpleTask:
    """Lightweight training task."""
    DEFAULT_CPI = 2.0

    def __init__(self, uid, wk=None, cpi=None):
        self.id = uid
        self.wk = wk if wk is not None else random.uniform(*WORKLOAD_RANGE)
        self.cpi = cpi if cpi is not None else max(0.4, self.DEFAULT_CPI + random.uniform(-0.3, 0.3))
        self.deadline = DEADLINE
        self.base_val = 100.0
        self.tau_k = calc_tau(self.base_val, REVENUE_COST, REVENUE_DELTA, REVENUE_GAMMA)


def calc_cpi_efficiency(cpi):

    eta_max = CPI_MODEL_PARAMS['eta_max']
    eta_min = CPI_MODEL_PARAMS['eta_min']
    cpi_ref = CPI_MODEL_PARAMS['cpi_ref']
    k = CPI_MODEL_PARAMS['k']
    return (eta_max - eta_min) / (1 + np.exp(k * (cpi - cpi_ref))) + eta_min



BETA_1 = 0.3
BETA_2 = 0.5
BETA_3 = 0.1
ALPHA_MAP = {'High': 0.2, 'Mid': 0.45, 'Low': 0.8}

def calc_real_time_train(task, agent, concurrency, omega):

    eta = calc_cpi_efficiency(task.cpi)
    dynamic_output = agent.capacity * eta
    base_t = task.wk / dynamic_output
    alpha = ALPHA_MAP.get(getattr(agent, 'type', 'Mid'), 0.45)
    f = BETA_1 * omega + BETA_2 * omega**2 + BETA_3 * max(0, concurrency - 1)
    return base_t * (1 + alpha * f)


def calc_revenue(task, agent, concurrency, omega=None):

    import math
    if omega is None:
        omega = agent.current_load

    real_t = calc_real_time_train(task, agent, concurrency, omega)
    effective_deadline = task.deadline - task.tau_k

    x = REVENUE_DELTA * (real_t - effective_deadline)
    sig = 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

    utility = ((1.0 - sig) ** REVENUE_GAMMA) * task.base_val - REVENUE_COST
    return utility


def create_agents(n_agents):

    agents = []
    uid = 0
    for atype, info in AGENT_DISTRIBUTION.items():
        count = max(1, int(n_agents * info['ratio']))
        for _ in range(count):
            if uid >= n_agents:
                break
            cap = random.uniform(*info['cap_range'])
            agents.append(SimpleAgent(uid, atype, cap))
            uid += 1

    while len(agents) < n_agents:
        cap = random.uniform(50, 89)
        agents.append(SimpleAgent(uid, 'Mid', cap))
        uid += 1
    return agents[:n_agents]


def create_tasks(n_tasks):





    tasks = []
    for i in range(n_tasks):
        wk = random.uniform(*WORKLOAD_RANGE)

        cpi = max(0.4, min(8.0, np.random.lognormal(mean=0.65, sigma=0.45)))
        tasks.append(SimpleTask(i, wk, cpi))
    return tasks



# Gym-style scheduling environment

class TaskSchedulingEnv:
    """Scheduling environment for DDQN training."""
















    def __init__(self, n_agents=AGENT_COUNT):
        self.n_agents = n_agents
        self.state_dim = 2 + 2 * n_agents
        self.action_dim = n_agents

    def reset(self, n_tasks=None):

        if n_tasks is None:
            n_tasks = random.choice([60, 80, 100, 120])
        self.agents = create_agents(self.n_agents)
        self.tasks = sorted(create_tasks(n_tasks), key=lambda t: t.wk)
        self.current_idx = 0
        self.total_revenue = 0
        self.success_count = 0
        return self._get_state()

    def _get_state(self):

        if self.current_idx >= len(self.tasks):
            return np.zeros(self.state_dim, dtype=np.float32)

        task = self.tasks[self.current_idx]
        state = [task.wk / 80.0, task.cpi / 4.0]
        for ag in self.agents:
            state.append(ag.current_load)
            state.append(ag.capacity / 120.0)
        return np.array(state, dtype=np.float32)

    def get_feasible_mask(self):

        if self.current_idx >= len(self.tasks):
            return np.zeros(self.n_agents, dtype=bool)

        task = self.tasks[self.current_idx]
        mask = np.zeros(self.n_agents, dtype=bool)
        for i, ag in enumerate(self.agents):
            omega = task.wk / ag.capacity
            if ag.current_load + omega <= 1.0:
                mask[i] = True
        return mask

    def step(self, action):

        task = self.tasks[self.current_idx]
        agent = self.agents[action]

        omega = task.wk / agent.capacity
        if agent.current_load + omega <= 1.0:

            agent.current_load += omega
            agent.concurrent_tasks.append(task)
            n = len(agent.concurrent_tasks)


            real_t = calc_real_time_train(task, agent, n, agent.current_load)

            if real_t <= task.deadline:

                self.success_count += 1
                self.total_revenue += task.base_val
                margin = (task.deadline - real_t) / task.deadline
                reward = 5.0 + 5.0 * margin
            else:

                overtime = (real_t - task.deadline) / task.deadline
                reward = -5.0 * min(overtime, 2.0)
        else:

            reward = -5.0

        self.current_idx += 1
        done = self.current_idx >= len(self.tasks)
        next_state = self._get_state()

        return next_state, reward, done



# DDQN model


class DDQNNetwork(nn.Module):
    """Two-layer DDQN network."""







    def __init__(self, state_dim, action_dim, hidden_size=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.net(x)



# Replay buffer

class ReplayBuffer:
    """Fixed-size replay buffer."""
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(masks),
            np.array(next_masks)
        )

    def __len__(self):
        return len(self.buffer)



# Federated trainer

class FLDRLTrainer:
    """Federated DDQN trainer."""









    def __init__(self, n_agents=AGENT_COUNT,
                 n_clients=5,
                 global_rounds=300,
                 local_episodes=10,
                 local_train_steps=150,
                 clients_per_round=5,
                 lr=0.001,
                 gamma=0.9,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_rounds=240,
                 batch_size=64,
                 target_update=200,
                 buffer_capacity=20000):

        self.n_agents = n_agents
        self.n_clients = n_clients
        self.global_rounds = global_rounds
        self.local_episodes = local_episodes
        self.local_train_steps = local_train_steps
        self.clients_per_round = clients_per_round
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma


        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_rounds


        self.env = TaskSchedulingEnv(n_agents)
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.global_model = DDQNNetwork(state_dim, action_dim).to(self.device)


        self.client_models = []
        self.client_target_models = []
        self.client_buffers = []
        self.client_optimizers = []
        self.client_step_counts = []

        for _ in range(n_clients):
            current_net = DDQNNetwork(state_dim, action_dim).to(self.device)
            target_net = DDQNNetwork(state_dim, action_dim).to(self.device)

            current_net.load_state_dict(self.global_model.state_dict())
            target_net.load_state_dict(self.global_model.state_dict())

            self.client_models.append(current_net)
            self.client_target_models.append(target_net)
            self.client_buffers.append(ReplayBuffer(buffer_capacity))
            self.client_optimizers.append(optim.Adam(current_net.parameters(), lr=lr))
            self.client_step_counts.append(0)


        self.best_reward = -float('inf')

    def collect_experience(self, client_idx):






        model = self.client_models[client_idx]
        buffer = self.client_buffers[client_idx]
        total_reward = 0
        total_success = 0
        total_tasks = 0

        for _ in range(self.local_episodes):
            state = self.env.reset()
            done = False

            while not done:
                mask = self.env.get_feasible_mask()
                feasible_actions = np.where(mask)[0]


                if len(feasible_actions) == 0:

                    action = random.randint(0, self.n_agents - 1)
                elif random.random() < self.epsilon:

                    action = int(random.choice(feasible_actions))
                else:

                    with torch.no_grad():
                        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = model(state_t).cpu().numpy()[0]
                        q_values[~mask] = -1e9
                        action = int(np.argmax(q_values))

                next_state, reward, done = self.env.step(action)
                next_mask = self.env.get_feasible_mask() if not done else np.zeros(self.n_agents, dtype=bool)

                buffer.push(state, action, reward, next_state, done, mask, next_mask)
                state = next_state
                total_reward += reward

            total_success += self.env.success_count
            total_tasks += len(self.env.tasks)

        return total_reward / self.local_episodes, total_success / max(total_tasks, 1)

    def local_train(self, client_idx):










        model = self.client_models[client_idx]
        target_model = self.client_target_models[client_idx]
        optimizer = self.client_optimizers[client_idx]
        buffer = self.client_buffers[client_idx]

        if len(buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0
        steps = min(self.local_train_steps, len(buffer) // self.batch_size)

        for _ in range(steps):

            states, actions, rewards, next_states, dones, masks, next_masks = \
                buffer.sample(self.batch_size)

            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            dones_t = torch.FloatTensor(dones).to(self.device)
            next_masks_t = torch.BoolTensor(next_masks).to(self.device)

            # DDQN update


            q_values = model(states_t)
            q_value = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():

                next_q_current = model(next_states_t)
                next_q_current[~next_masks_t] = -1e9
                next_actions = next_q_current.argmax(dim=1)


                next_q_target = target_model(next_states_t)
                next_q_value = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)


                target = rewards_t + self.gamma * next_q_value * (1.0 - dones_t)


            loss = nn.MSELoss()(q_value, target)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()


            self.client_step_counts[client_idx] += 1
            if self.client_step_counts[client_idx] % self.target_update == 0:
                target_model.load_state_dict(model.state_dict())

        return total_loss / max(steps, 1)

    def fedavg_aggregate(self, selected_clients):






        total_samples = sum(len(self.client_buffers[c]) for c in selected_clients)
        if total_samples == 0:
            return


        global_dict = {}
        for key in self.global_model.state_dict():
            global_dict[key] = torch.zeros_like(self.global_model.state_dict()[key])
            for c in selected_clients:
                weight = len(self.client_buffers[c]) / total_samples
                global_dict[key] += weight * self.client_models[c].state_dict()[key]

        self.global_model.load_state_dict(global_dict)


        for i in range(self.n_clients):
            self.client_models[i].load_state_dict(global_dict)

    def train(self):








        print(f"{'='*60}")
        print(f"  FL-DRL (FL-DDQN) 训练")
        print(f"{'='*60}")
        print(f"  设备:           {self.device}")
        print(f"  智能体数:       {self.n_agents}")
        print(f"  状态维度:       {self.env.state_dim}")
        print(f"  动作维度:       {self.env.action_dim}")
        print(f"  全局轮次:       {self.global_rounds}")
        print(f"  联邦客户端:     {self.n_clients} (每轮参与 {self.clients_per_round})")
        print(f"  本地episodes:   {self.local_episodes}")
        print(f"  本地训练步:     {self.local_train_steps}")
        print(f"{'='*60}\n")

        for round_idx in range(self.global_rounds):

            selected = random.sample(range(self.n_clients), self.clients_per_round)

            round_rewards = []
            round_losses = []
            round_success_rates = []


            for c in selected:

                avg_reward, success_rate = self.collect_experience(c)
                round_rewards.append(avg_reward)
                round_success_rates.append(success_rate)


                avg_loss = self.local_train(c)
                round_losses.append(avg_loss)


            self.fedavg_aggregate(selected)


            self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay_rate)


            avg_r = np.mean(round_rewards)
            avg_l = np.mean(round_losses)
            avg_sr = np.mean(round_success_rates)

            if avg_r > self.best_reward:
                self.best_reward = avg_r
                self.save_model(os.path.join('models', 'fl_drl_model_best.pt'))

            if (round_idx + 1) % 5 == 0 or round_idx == 0:
                print(f"  Round {round_idx+1:4d}/{self.global_rounds} | "
                      f"Reward: {avg_r:8.1f} | Loss: {avg_l:.4f} | "
                      f"Success: {avg_sr:.1%} | ε: {self.epsilon:.3f}")


        self.save_model(os.path.join('models', 'fl_drl_model.pt'))
        print(f"\n{'='*60}")
        print(f"  ✅ 训练完成！")
        print(f"  最终模型: models/fl_drl_model.pt")
        print(f"  最佳模型: models/fl_drl_model_best.pt")
        print(f"  最佳奖励: {self.best_reward:.1f}")
        print(f"{'='*60}")

    def save_model(self, path):

        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'state_dim': self.env.state_dim,
            'action_dim': self.env.action_dim,
            'n_agents': self.n_agents,
            'hidden_size': 200,
        }, path)



# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FL-DRL (FL-DDQN) 训练脚本')
    parser.add_argument('--agents', type=int, default=40,
                        help='智能体数量 (默认: 40)')
    parser.add_argument('--rounds', type=int, default=300,
                        help='全局联邦训练轮次 (默认: 300)')
    parser.add_argument('--clients', type=int, default=5,
                        help='联邦客户端总数 (默认: 5)')
    parser.add_argument('--clients-per-round', type=int, default=5,
                        help='每轮参与的客户端数 (默认: 5)')
    parser.add_argument('--local-episodes', type=int, default=10,
                        help='每轮每客户端采集的episode数 (默认: 10)')
    parser.add_argument('--local-steps', type=int, default=150,
                        help='每轮本地DDQN训练步数 (默认: 150)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    args = parser.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


    trainer = FLDRLTrainer(
        n_agents=args.agents,
        n_clients=args.clients,
        global_rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        local_episodes=args.local_episodes,
        local_train_steps=args.local_steps,
    )
    trainer.train()
