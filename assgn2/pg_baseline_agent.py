import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
import os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs('results/pg_baseline', exist_ok=True)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[64, 64]):
        super(PolicyNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=[64, 64]):
        super(ValueNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Return scalar value (squeeze last dimension)
        return self.network(x).squeeze()

class PGBaselineAgent:
    def __init__(self, state_dim, action_dim, lr_policy=1e-3, lr_value=1e-3, gamma=0.99):
        self.gamma = gamma
        
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        
        self.value = ValueNetwork(state_dim)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr_value)
        
        self.rewards = []
        self.log_probs = []
        self.values = []
    
    def select_action(self, state):
        action, log_prob = self.policy.get_action(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        value = self.value(state_tensor)
        return action, log_prob, value
    
    def store_transition(self, reward, log_prob, value):
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_returns(self):
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns)
    
    def update(self):
        if len(self.rewards) == 0:
            return 0.0, 0.0
        
        returns = self.compute_returns()
        values = torch.stack(self.values)
        
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        value_loss = F.mse_loss(values, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        policy_loss_value = policy_loss.item()
        value_loss_value = value_loss.item()
        
        self.clear_memory()
        
        return policy_loss_value, value_loss_value
    
    def clear_memory(self):
        self.rewards = []
        self.log_probs = []
        self.values = []

def train_pg_baseline(num_episodes=500, save_name="pg_baseline"):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PGBaselineAgent(state_dim, action_dim)
    
    episode_rewards = []
    avg_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        done = False
        
        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(reward, log_prob, value)
            
            state = next_state
            episode_reward += reward
        
        agent.update()
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        avg_rewards.append(avg_reward)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {avg_reward:.2f}")
        
        if len(episode_rewards) >= 10 and avg_reward >= 180:
            print(f"Solved in {episode + 1} episodes! Average reward: {avg_reward:.2f}")
            break
    
    env.close()
    
    torch.save({
        'policy': agent.policy.state_dict(),
        'value': agent.value.state_dict(),
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards
    }, f'results/pg_baseline/{save_name}.pth')
    
    # Save results as JSON for report generation
    with open('results/pg_baseline_results.json', 'w') as f:
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

def compare_pg_vs_baseline():
    from pg_agent import train_reinforce
    
    print("=" * 60)
    print("Training Vanilla REINFORCE")
    print("=" * 60)
    pg_rewards, pg_avg, _ = train_reinforce(num_episodes=300, save_name="vanilla_pg")
    
    print("\n" + "=" * 60)
    print("Training PG with Baseline")
    print("=" * 60)
    baseline_rewards, baseline_avg, _ = train_pg_baseline(num_episodes=300, 
                                                          save_name="with_baseline")
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(pg_rewards, alpha=0.3, label='REINFORCE', color='blue')
    plt.plot(baseline_rewards, alpha=0.3, label='PG+Baseline', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Raw Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(pg_avg, label='REINFORCE', linewidth=2, color='blue')
    plt.plot(baseline_avg, label='PG+Baseline', linewidth=2, color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 10)')
    plt.title('Average Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    window = 20
    pg_var = [np.var(pg_rewards[max(0, i-window):i+1]) for i in range(len(pg_rewards))]
    baseline_var = [np.var(baseline_rewards[max(0, i-window):i+1]) 
                    for i in range(len(baseline_rewards))]
    
    plt.plot(pg_var, label='REINFORCE', alpha=0.7, color='blue')
    plt.plot(baseline_var, label='PG+Baseline', alpha=0.7, color='green')
    plt.xlabel('Episode')
    plt.ylabel(f'Variance (window={window})')
    plt.title('Variance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/pg_baseline/comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return pg_rewards, pg_avg, baseline_rewards, baseline_avg

if __name__ == "__main__":
    print("Training PG with Baseline on CartPole-v1")
    rewards, avg_rewards, agent = train_pg_baseline()
    plot_training_curve(rewards, avg_rewards, "PG with Baseline Training Curve", 
                       "results/pg_baseline/training_curve.png")
    
    print("\nComparing with vanilla REINFORCE...")
    compare_pg_vs_baseline()
    print("\nPG with Baseline training complete!")