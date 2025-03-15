import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 定义Actor和Critic网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

# PPO算法实现
class PPO:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.actor(state)
        return action.item()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        values = self.critic(states)
        next_values = self.critic(next_states)
        target_values = rewards + self.gamma * next_values * (1 - dones)
        advantage = target_values - values

        actor_loss = -torch.mean(advantage * self.actor(states))
        critic_loss = torch.mean(advantage ** 2)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

# 主训练过程
def main():
    env = gym.make("Pendulum-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim, action_dim)

    episodes = 1000
    for episode in range(episodes):
        state, info = env.reset()
        done = False
        rewards = 0.0
        states, actions, rewards_list, next_states, dones = [], [], [], [], []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step([action])
            env.render()

            states.append(state)
            actions.append(action)
            rewards_list.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            rewards += reward

        agent.update(states, actions, rewards_list, next_states, dones)
        print(f"Episode: {episode}, Total Reward: {rewards}")

    env.close()

if __name__ == "__main__":
    main()