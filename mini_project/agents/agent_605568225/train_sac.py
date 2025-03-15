"""
Training script for SAC (Soft Actor-Critic) on MetaDrive environment.

Usage:
    python train_sac.py 
"""
import argparse
import datetime
import os
import pathlib
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import environment
from env import get_training_env, get_validation_env
from utils import make_envs, step_envs

# Set the path to the current folder
FOLDER_ROOT = pathlib.Path(__file__).parent


class ReplayBuffer:
    """Simple replay buffer for storing experience tuples."""
    
    def __init__(self, buffer_size, state_dim, action_dim, device):
        """Initialize a ReplayBuffer object.
        
        Args:
            buffer_size (int): Maximum size of buffer
            state_dim (int): Dimension of state
            action_dim (int): Dimension of action
            device: Torch device
        """
        self.buffer_size = buffer_size
        self.device = device
        self.position = 0
        self.size = 0
        
        # Initialize buffers
        self.state = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.action = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_state = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.done = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.state[self.position] = state
        self.action[self.position] = action
        self.reward[self.position] = reward
        self.next_state[self.position] = next_state
        self.done[self.position] = done
        
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        states = torch.FloatTensor(self.state[indices]).to(self.device)
        actions = torch.FloatTensor(self.action[indices]).to(self.device)
        rewards = torch.FloatTensor(self.reward[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_state[indices]).to(self.device)
        dones = torch.FloatTensor(self.done[indices]).to(self.device)
        
        return states, actions, rewards, next_states, dones


class Actor(nn.Module):
    """Actor network for SAC that maps states to a distribution over actions."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mu, log_std
    
    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from the Gaussian distribution
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        
        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        
        # Calculate log probability of the action
        log_prob = normal.log_prob(x_t)
        
        # Apply the change of variables formula for the tanh transform
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob


class Critic(nn.Module):
    """Critic network for SAC that maps (state, action) pairs to Q-values."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        # Q1 value
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        # Q2 value
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        """Return only Q1 value for deterministic action selection."""
        x = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        return q1


class SAC:
    """Soft Actor-Critic implementation."""
    
    def __init__(self, state_dim, action_dim, device,
                 hidden_dim=256,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 tau=0.005,
                 gamma=0.99,
                 alpha=0.2,
                 auto_entropy_tuning=True):
        """Initialize SAC parameters.
        
        Args:
            state_dim (int): Dimension of state
            action_dim (int): Dimension of action
            device: Torch device
            hidden_dim (int): Hidden layer dimension
            lr_actor (float): Learning rate for actor
            lr_critic (float): Learning rate for critic
            tau (float): Soft target update parameter
            gamma (float): Discount factor
            alpha (float): Temperature parameter for entropy
            auto_entropy_tuning (bool): Whether to automatically tune alpha
        """
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        
        # Initialize actor network
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Initialize critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize target critic networks with the same weights
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Entropy adjustment
        if auto_entropy_tuning:
            # Target entropy is -dim(A)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
    
    def select_action(self, state, evaluate=False):
        """Select an action from the policy."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate:
            # Use mean action for evaluation
            with torch.no_grad():
                mu, _ = self.actor.forward(state)
                return torch.tanh(mu).cpu().numpy().flatten()
        else:
            # Sample from the distribution during training
            with torch.no_grad():
                action, _ = self.actor.sample(state)
                return action.cpu().numpy().flatten()
    
    def update_parameters(self, replay_buffer, batch_size):
        """Update the networks parameters."""
        # Sample a batch of transitions from replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Update critic networks
        with torch.no_grad():
            # Sample actions from the target policy
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # Target Q-values
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor network
        actions_new, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        
        # Compute actor loss (with entropy regularization)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Adjust entropy coefficient if needed
        alpha_loss = None
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update of target critic network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if alpha_loss is not None else 0,
            'alpha': self.alpha.item() if hasattr(self.alpha, 'item') else self.alpha
        }
    
    def save(self, filepath):
        """Save model parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        if self.auto_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()
        
        print(f"Model loaded from {filepath}")


def evaluate_policy(agent, env, num_episodes=10):
    """Evaluate the agent on the environment."""
    avg_reward = 0.
    avg_completion = 0.
    success_rate = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        avg_reward += episode_reward
        if 'route_completion' in info:
            avg_completion += info['route_completion']
        if 'arrive_dest' in info and info['arrive_dest']:
            success_rate += 1
    
    avg_reward /= num_episodes
    avg_completion /= num_episodes
    success_rate /= num_episodes
    
    return avg_reward, avg_completion, success_rate


def train_sac():
    """Main training function for SAC."""
    # Initialize training parameters
    state_dim = 259  # MetaDrive observation dimension
    action_dim = 2   # MetaDrive action dimension (steering, acceleration)
    hidden_dim = 256
    buffer_size = 1_000_000
    batch_size = 256
    num_envs = 1  # Number of parallel environments
    updates_per_step = 1
    start_steps = 10000  # Random exploration steps
    max_steps = 1_000_000  # Total environment steps
    eval_freq = 5000  # Evaluation frequency
    eval_episodes = 10  # Number of evaluation episodes
    save_freq = 50000  # Model saving frequency
    
    # Create log directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = FOLDER_ROOT / f"sac_logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up TensorBoard logger
    writer = SummaryWriter(log_dir=log_dir)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environments
    env = get_training_env()
    eval_env = get_validation_env()
    
    # Initialize SAC agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden_dim=hidden_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        tau=0.005,
        gamma=0.99,
        alpha=0.2,
        auto_entropy_tuning=True
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, device)
    
    # Initialize training variables
    total_steps = 0
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    
    # Initialize tracking metrics
    rewards_window = deque(maxlen=100)
    completion_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    
    # Get initial state
    state, _ = env.reset()
    
    print("Starting training...")
    training_start_time = time.time()
    
    while total_steps < max_steps:
        # Select action
        if total_steps < start_steps:
            # Random exploration
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            # Use SAC policy
            action = agent.select_action(state)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        # Update state and counters
        state = next_state
        episode_reward += reward
        episode_steps += 1
        total_steps += 1
        
        # Update networks if enough samples are available
        if total_steps > batch_size and total_steps > start_steps:
            for _ in range(updates_per_step):
                update_info = agent.update_parameters(replay_buffer, batch_size)
                writer.add_scalar('Loss/critic', update_info['critic_loss'], total_steps)
                writer.add_scalar('Loss/actor', update_info['actor_loss'], total_steps)
                writer.add_scalar('Loss/alpha', update_info['alpha_loss'], total_steps)
                writer.add_scalar('Param/alpha', update_info['alpha'], total_steps)
        
        # Evaluate the agent periodically
        if total_steps % eval_freq == 0:
            eval_reward, eval_completion, eval_success = evaluate_policy(agent, eval_env, eval_episodes)
            writer.add_scalar('Eval/reward', eval_reward, total_steps)
            writer.add_scalar('Eval/completion', eval_completion, total_steps)
            writer.add_scalar('Eval/success_rate', eval_success, total_steps)
            
            print(f"Evaluation at step {total_steps}: "
                  f"Reward: {eval_reward:.2f}, Completion: {eval_completion:.2f}, "
                  f"Success Rate: {eval_success:.2f}")
        
        # Save model periodically
        if total_steps % save_freq == 0:
            agent.save(log_dir / f"sac_model_step_{total_steps}.pt")
        
        # Reset environment if episode is done
        if done:
            # Track metrics
            rewards_window.append(episode_reward)
            if 'route_completion' in info:
                completion_window.append(info['route_completion'])
            if 'arrive_dest' in info:
                success_window.append(float(info['arrive_dest']))
            
            # Log episode stats
            writer.add_scalar('Train/reward', episode_reward, total_steps)
            writer.add_scalar('Train/episode_length', episode_steps, total_steps)
            if 'route_completion' in info:
                writer.add_scalar('Train/completion', info['route_completion'], total_steps)
            if 'arrive_dest' in info:
                writer.add_scalar('Train/success', float(info['arrive_dest']), total_steps)
            
            # Log moving averages every 10 episodes
            if episode_count % 10 == 0:
                writer.add_scalar('Train/avg_reward', np.mean(rewards_window), total_steps)
                if completion_window:
                    writer.add_scalar('Train/avg_completion', np.mean(completion_window), total_steps)
                if success_window:
                    writer.add_scalar('Train/avg_success', np.mean(success_window), total_steps)
                
                # Print progress
                hours, remainder = divmod(time.time() - training_start_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Episode {episode_count}, Step {total_steps}: "
                      f"Avg Reward: {np.mean(rewards_window):.2f}, "
                      f"Avg Completion: {np.mean(completion_window) if completion_window else 0:.2f}, "
                      f"Avg Success: {np.mean(success_window) if success_window else 0:.2f}, "
                      f"Time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Reset for new episode
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_count += 1
    
    # Save final model
    agent.save(FOLDER_ROOT / "sac_model.pt")
    print("Training completed! Final model saved.")
    
    # Clean up
    env.close()
    eval_env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent for MetaDrive")
    args = parser.parse_args()
    
    train_sac()