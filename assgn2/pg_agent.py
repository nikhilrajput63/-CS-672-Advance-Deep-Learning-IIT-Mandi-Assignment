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

os.makedirs('results/pg', exist_ok=True)

class PolicyNetwork(nn.Module):
    """
    Policy network for discrete actions (Categorical distribution)
    For CartPole: outputs action probabilities for left/right
    """
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
        """Sample action from categorical distribution"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class REINFORCEAgent:
    """
    REINFORCE: Monte Carlo Policy Gradient
    - Collects full episode trajectory
    - Computes returns-to-go G_t = sum(gamma^k * r_{t+k})
    - Updates policy using: gradient(log(pi(a|s)) * G_t)
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.rewards = []
        self.log_probs = []
    
    def select_action(self, state):
        """Sample action from current policy"""
        action, log_prob = self.policy.get_action(state)
        return action, log_prob
    
    def store_transition(self, reward, log_prob):
        """Store reward and log prob for current timestep"""
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
    
    def compute_returns(self):
        """
        Compute returns-to-go (Monte Carlo returns)
        G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ...
        """
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        # Normalize for variance reduction
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update_policy(self):
        """
        REINFORCE update rule:
        theta <- theta + alpha * gradient(log(pi(a|s)) * G_t)
        """
        if len(self.rewards) == 0:
            return 0.0
        
        # Compute returns-to-go for entire episode
        returns = self.compute_returns()
        
        # Policy gradient: -log(pi) * G (negative for gradient ascent via descent)
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Gradient ascent step (via optimizer minimizing negative)
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        loss_value = policy_loss.item()
        self.clear_memory()
        
        return loss_value
    
    def clear_memory(self):
        """Clear episode buffer"""
        self.rewards = []
        self.log_probs = []

def train_reinforce(num_episodes=500, save_name="reinforce"):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(state_dim, action_dim)
    
    episode_rewards = []
    avg_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + episode)
        episode_reward = 0
        done = False
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(reward, log_prob)
            
            state = next_state
            episode_reward += reward
        
        agent.update_policy()
        
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
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards
    }, f'results/pg/{save_name}.pth')
    
    # Save results as JSON for report generation
    with open('results/pg_results.json', 'w') as f:
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

def analyze_variance(num_runs=5):
    print("\n" + "=" * 60)
    print("Analyzing Gradient Variance Across Multiple Runs")
    print("=" * 60)
    
    all_rewards = []
    all_avg_rewards = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        rewards, avg_rewards, _ = train_reinforce(num_episodes=200, 
                                                  save_name=f"run_{run}")
        all_rewards.append(rewards)
        all_avg_rewards.append(avg_rewards)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, alpha=0.3, label=f'Run {i+1}' if i < 3 else '')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('REINFORCE: Variance Across Runs')
    if num_runs <= 3:
        plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    max_len = max(len(r) for r in all_avg_rewards)
    padded = [list(r) + [r[-1]]*(max_len-len(r)) for r in all_avg_rewards]
    
    mean_rewards = np.mean(padded, axis=0)
    std_rewards = np.std(padded, axis=0)
    
    episodes = np.arange(len(mean_rewards))
    plt.plot(episodes, mean_rewards, label='Mean', linewidth=2)
    plt.fill_between(episodes, mean_rewards - std_rewards, 
                     mean_rewards + std_rewards, alpha=0.3, label='±1 Std')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('REINFORCE: Mean ± Std')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/pg/variance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return all_rewards, all_avg_rewards

if __name__ == "__main__":
    print("Training REINFORCE Agent on CartPole-v1")
    print("=" * 60)
    print("\nREINFORCE Algorithm:")
    print("  - Monte Carlo Policy Gradient")
    print("  - Returns-to-go: G_t = sum(gamma^k * r_{t+k})")
    print("  - Gradient: nabla(log(pi(a|s)) * G_t)")
    print("  - Categorical policy (discrete actions)")
    print("=" * 60)
    
    rewards, avg_rewards, agent = train_reinforce()
    plot_training_curve(rewards, avg_rewards, "REINFORCE Training Curve", 
                       "results/pg/training_curve.png")
    
    print("\nAnalyzing variance...")
    analyze_variance()
     
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    print("""
1. REINFORCE uses Monte Carlo returns (full episode)
   - G_t = R_{t+1} + gamma*R_{t+2} + gamma^2*R_{t+3} + ...
   
2. High variance because:
   - Single trajectory sample per update
   - Full episode return has many random factors
   - No bootstrapping (unlike TD methods)
   
3. Credit assignment problem:
   - All actions get same return signal
   - Hard to determine which actions were crucial
   
4. Why normalize returns?
   - Reduces variance without adding bias
   - Stabilizes training across different scales
   
5. Gradient estimator:
   - Unbiased but high variance
   - Needs many samples to converge
   - Solution: Use baseline (Part III)
""")
    print("\nREINFORCE training complete!")