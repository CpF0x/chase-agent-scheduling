"""Run FL-DRL model inference."""








import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
torch.set_num_threads(1)
import torch.nn as nn
import numpy as np


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


class FLDRLInference:
    """Inference wrapper for the trained FL-DRL model."""





    def __init__(self, model_path, device='cpu'):





        self.device = torch.device(device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"请先在 GPU 上运行 drl_train.py 训练模型，\n"
                f"然后将 fl_drl_model.pt 下载到此目录。"
            )

        # Load trained model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dim = checkpoint['state_dim']
        action_dim = checkpoint['action_dim']
        hidden_size = checkpoint.get('hidden_size', 200)
        self.n_agents = checkpoint['n_agents']

        self.model = DDQNNetwork(state_dim, action_dim, hidden_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"  [FL-DRL] 模型加载成功: {model_path}")
        print(f"  [FL-DRL] 状态维度={state_dim}, 动作维度={action_dim}, 智能体数={self.n_agents}")

    def build_state(self, task, agents):










        state = [task.wk / 80.0, task.cpi / 4.0]
        for ag in agents:
            state.append(ag.current_load)
            state.append(ag.capacity / 120.0)
        return np.array(state, dtype=np.float32)

    def choose_action(self, state, feasible_mask):













        feasible_actions = np.where(feasible_mask)[0]
        if len(feasible_actions) == 0:
            return None

        # Fallback on model mismatch
        n_agents_actual = len(feasible_mask)
        if n_agents_actual != self.n_agents:
            return int(np.random.choice(feasible_actions))

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_t).cpu().numpy()[0]

            q_values[~feasible_mask] = -1e9
            return int(np.argmax(q_values))

    def choose_action_stochastic(self, state, feasible_mask, temperature=1.0):














        feasible_actions = np.where(feasible_mask)[0]
        if len(feasible_actions) == 0:
            return None


        n_agents_actual = len(feasible_mask)
        if n_agents_actual != self.n_agents:
            return int(np.random.choice(feasible_actions))

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_t).cpu().numpy()[0]
            # Sample feasible actions
            q_feasible = q_values[feasible_actions].astype(np.float64)
            q_feasible -= q_feasible.max()
            probs = np.exp(q_feasible / temperature)
            probs /= probs.sum()
            chosen_idx = np.random.choice(len(feasible_actions), p=probs)
            return int(feasible_actions[chosen_idx])
