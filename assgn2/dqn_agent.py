import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import json
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs('results/dqn', exist_ok=True)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update=10,
                 use_target_network=True):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_target_network = use_target_network
        
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def select_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.use_target_network:
                next_q_values = self.target_network(next_states).max(1)[0]
            else:
                next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

def train_dqn(num_episodes=500, use_target_network=True, buffer_size=10000, 
              save_name="dqn"):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, use_target_network=use_target_network,
                     buffer_size=buffer_size)
    
    episode_rewards = []
    avg_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
        
        if use_target_network and episode % agent.target_update == 0:
            agent.update_target_network()
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        avg_rewards.append(avg_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        if len(episode_rewards) >= 10 and avg_reward >= 180:
            print(f"Solved in {episode + 1} episodes! Average reward: {avg_reward:.2f}")
            break
    
    env.close()
    
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards
    }, f'results/dqn/{save_name}.pth')
    
    # Save results as JSON for report generation
    with open('results/dqn_results.json', 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'avg_rewards': avg_rewards,
            'episodes_trained': len(episode_rewards)
        }, f)
    
    return episode_rewards, avg_rewards, agent

def plot_training_curve(rewards, avg_rewards, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (last 10)', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_configurations():
    print("=" * 60)
    print("Training DQN WITH Target Network")
    print("=" * 60)
    rewards_with, avg_with, _ = train_dqn(use_target_network=True, save_name="with_target")
    
    print("\n" + "=" * 60)
    print("Training DQN WITHOUT Target Network")
    print("=" * 60)
    rewards_without, avg_without, _ = train_dqn(use_target_network=False, save_name="without_target")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(avg_with, label='With Target Network', linewidth=2)
    plt.plot(avg_without, label='Without Target Network', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 10)')
    plt.title('DQN: Target Network Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(rewards_with, alpha=0.3, label='With Target Network')
    plt.plot(rewards_without, alpha=0.3, label='Without Target Network')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('DQN: Raw Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/dqn/target_network_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return rewards_with, avg_with, rewards_without, avg_without

if __name__ == "__main__":
    print("Training DQN Agent on CartPole-v1")
    rewards, avg_rewards, agent = train_dqn()
    plot_training_curve(rewards, avg_rewards, "DQN Training Curve", 
                       "results/dqn/training_curve.png")
    
    print("\nRunning Target Network Comparison...")
    compare_configurations()
    print("\nDQN training complete!")