"""
train_ddpg.py

Trains a DDPG agent on ParkingEnv (autonomous_parking_env_with_collision.py):
– State: [x₀, y₀, α] (geometric center + heading)
– Action: [s, β] (continuous speed & steering)
– Reward: R = 10[(1−w)R_d + w R_α] + R_p (Li & Long 2021, Eqs. 4–6) :contentReference[oaicite:0]{index=0}
– Logs total reward, steps, actor & critic losses every `log_interval` episodes
"""
import sys
import os

import numpy as np
import math, random
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
torch.autograd.set_detect_anomaly(True)


# Import the Gym‐style parking environment
from autonomous_parking_env import ParkingEnv, MAX_DISPLACEMENT, MAX_STEER


# — Replay Buffer —
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(tuple(args))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))
    def __len__(self):
        return len(self.buffer)

# — Actor & Critic Networks (Li & Long §3.4) :contentReference[oaicite:1]{index=1} —
class Actor(nn.Module):
    def __init__(self, S, A):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S, 30), nn.Tanh(),
            nn.Linear(30,45), nn.ReLU(),
            nn.Linear(45,20), nn.Tanh(),
            nn.Linear(20,A),  nn.Tanh()
        )
    def forward(self, s):
        return self.net(s)

class Critic(nn.Module):
    def __init__(self, S, A):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(S+A,45), nn.ReLU(),
            nn.Linear(45,30),  nn.ReLU(),
            nn.Linear(30,25),  nn.ReLU(),
            nn.Linear(25,1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s,a], dim=1))

# — Ornstein‐Uhlenbeck Noise (exploration) —
class OUNoise:
    def __init__(self, dim, mu=0.0, theta=0.15, sigma=0.2):
        self.state = np.ones(dim)*mu
        self.mu, self.theta, self.sigma = mu, theta, sigma
    def sample(self):
        dx = self.theta*(self.mu - self.state) + self.sigma*np.random.randn(len(self.state))
        self.state += dx
        return self.state

# — DDPG Agent —
class DDPG:
    def __init__(self, S, A, env, w=0.5):
        self.env     = env
        self.w       = w
        self.actor   = Actor(S,A)
        self.actor_t = Actor(S,A)
        self.critic  = Critic(S,A)
        self.critic_t= Critic(S,A)
        # copy weights
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        # optimizers (lr=0.002) :contentReference[oaicite:2]{index=2}
        self.a_opt = optim.Adam(self.actor.parameters(), lr=0.002)
        self.c_opt = optim.Adam(self.critic.parameters(),lr=0.002)
        # replay & noise
        self.buffer = ReplayBuffer(100_000)
        self.noise  = OUNoise(A)
        # hyperparams
        self.gamma = 0.92   # discount :contentReference[oaicite:3]{index=3}
        self.tau   = 0.01   # soft‐update :contentReference[oaicite:4]{index=4}
        self.batch = 140    # minibatch :contentReference[oaicite:5]{index=5}

    def select_action(self, state, explore=True):
        s_t = torch.FloatTensor(state).unsqueeze(0)
        a   = self.actor(s_t).detach().cpu().numpy()[0]
        if explore:
            a += self.noise.sample()
        # a[0]∈[-1,1] → speed s ∈ [MIN_DISPLACEMENT, MAX_DISPLACEMENT]
        # a[1]∈[-1,1] → steering β ∈ [−MAX_STEER, +MAX_STEER]
        s_cmd = a[0] * MAX_DISPLACEMENT
        b_cmd = a[1] * MAX_STEER
        return np.array([s_cmd, b_cmd], dtype=np.float32)


    def update(self):
        if len(self.buffer) < self.batch:
            return None, None

        # Sample a minibatch
        s, a, r, s2, d = self.buffer.sample(self.batch)
        s  = torch.FloatTensor(s)
        a  = torch.FloatTensor(a)
        r  = torch.FloatTensor(r).unsqueeze(1)
        s2 = torch.FloatTensor(s2)
        d  = torch.FloatTensor(d).unsqueeze(1)

        # ----- Critic update -----
        with torch.no_grad():
            a2 = self.actor_t(s2)
            q2 = self.critic_t(s2, a2)
            y  = r + self.gamma * (1 - d) * q2

        q      = self.critic(s, a)
        c_loss = nn.MSELoss()(q, y)

        self.c_opt.zero_grad()
        c_loss.backward()
        self.c_opt.step()

        # ----- Actor update -----
        a_pred = self.actor(s)
        a_loss = -self.critic(s, a_pred).mean()

        self.a_opt.zero_grad()
        a_loss.backward()
        self.a_opt.step()

        # ----- Soft‐update targets, outside autograd -----
        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(),  self.actor_t.parameters()):
                pt.copy_(pt * (1 - self.tau) + p * self.tau)
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.copy_(pt * (1 - self.tau) + p * self.tau)

        return a_loss.item(), c_loss.item()

# — Training Loop —
def train():
    env = ParkingEnv(px_per_m=100, w=0.5, max_steps=200)
    S, A = 3, 2
    agent = DDPG(S, A, env)

    num_episodes = 2000
    log_interval = 50

    for ep in range(1, num_episodes+1):
        state      = env.reset()
        ep_reward  = 0.0
        losses     = []
        for t in range(env.max_steps):
            action = agent.select_action(state, explore=True)
            next_s, reward, done, info = env.step(action)
            agent.buffer.push(state, action, reward, next_s, float(done))
            al, cl = agent.update()
            if al is not None:
                losses.append((al,cl))
            state = next_s
            ep_reward += reward
            if done:
                break

        if ep % log_interval == 0 or ep == 1:
            avg_al = np.mean([l[0] for l in losses]) if losses else 0.0
            avg_cl = np.mean([l[1] for l in losses]) if losses else 0.0
            print(f"Ep {ep:4d} | Reward {ep_reward:7.2f} | Steps {t:3d} | "
                  f"ActorLoss {avg_al:.4f} | CriticLoss {avg_cl:.4f}")

if __name__=="__main__":
    train()
