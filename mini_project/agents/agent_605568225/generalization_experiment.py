"""
Generalization experiment for SAC on MetaDrive environment.

This script trains and evaluates SAC agents on different numbers of training maps
to study the effect on generalization to unseen validation maps.

Usage:
    python generalization_experiment.py
"""
import argparse
import datetime
import os
import pathlib
import time
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

# Import environment and SAC implementation
from env import get_training_env, get_validation_env
from train_sac import SAC, ReplayBuffer, evaluate_policy

# Set the path to the current folder
FOLDER_ROOT = pathlib.Path(__file__).parent


def train_and_evaluate(num_train_maps, eval_interval=10000, total_steps=300000):
    """Train and periodically evaluate a SAC agent on a specific number of training maps."""
    # Training parameters
    state_dim = 259  # MetaDrive observation dimension
    action_dim = 2   # MetaDrive action dimension (steering, acceleration)
    hidden_dim = 256
    buffer_size = 500_000
    batch_size = 256
    updates_per_step = 1
    start_steps = 5000  # Random exploration steps
    
    # Set up log directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = FOLDER_ROOT / f"generalization_maps_{num_train_maps}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environments
    train_env = get_training_env({"num_scenarios": num_train_maps})
    eval_train_env = get_training_env({"num_scenarios": num_train_maps})
    validation_env = get_validation_env()
    
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
    total_training_steps = 0
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    
    # Initialize tracking metrics
    rewards_window = deque(maxlen=100)
    completion_window = deque(maxlen=100)
    success_window = deque(maxlen=100)
    
    # Evaluation tracking
    eval_results = {
        "train_steps": [],
        "train_rewards": [],
        "train_completions": [],
        "train_success_rates": [],
        "val_rewards": [],
        "val_completions": [],
        "val_success_rates": []
    }
    
    # Get initial state
    state, _ = train_env.reset()
    
    print(f"Starting training on {num_train_maps} maps...")
    training_start_time = time.time()
    
    while total_training_steps < total_steps:
        # Select action
        if total_training_steps < start_steps:
            # Random exploration
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            # Use SAC policy
            action = agent.select_action(state)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        
        # Store experience in replay buffer
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        # Update state and counters
        state = next_state
        episode_reward += reward
        episode_steps += 1
        total_training_steps += 1
        
        # Update networks if enough samples are available
        if total_training_steps > batch_size and total_training_steps > start_steps:
            for _ in range(updates_per_step):
                agent.update_parameters(replay_buffer, batch_size)
        
        # Evaluate the agent periodically
        if total_training_steps % eval_interval == 0:
            # Evaluate on training environment
            train_reward, train_completion, train_success = evaluate_policy(agent, eval_train_env, num_episodes=10)
            
            # Evaluate on validation environment
            val_reward, val_completion, val_success = evaluate_policy(agent, validation_env, num_episodes=10)
            
            # Record results
            eval_results["train_steps"].append(total_training_steps)
            eval_results["train_rewards"].append(train_reward)
            eval_results["train_completions"].append(train_completion)
            eval_results["train_success_rates"].append(train_success)
            eval_results["val_rewards"].append(val_reward)
            eval_results["val_completions"].append(val_completion)
            eval_results["val_success_rates"].append(val_success)
            
            # Save results to file
            with open(log_dir / "eval_results.json", "w") as f:
                json.dump(eval_results, f, indent=4)
            
            print(f"\nEvaluation at step {total_training_steps} with {num_train_maps} maps:")
            print(f"  Training Env: Reward: {train_reward:.2f}, Completion: {train_completion:.2f}, Success: {train_success:.2f}")
            print(f"  Validation Env: Reward: {val_reward:.2f}, Completion: {val_completion:.2f}, Success: {val_success:.2f}")
        
        # Reset environment if episode is done
        if done:
            # Track metrics
            rewards_window.append(episode_reward)
            if 'route_completion' in info:
                completion_window.append(info['route_completion'])
            if 'arrive_dest' in info:
                success_window.append(float(info['arrive_dest']))
            
            # Print progress every 10 episodes
            if episode_count % 10 == 0:
                avg_reward = np.mean(rewards_window) if rewards_window else 0
                avg_completion = np.mean(completion_window) if completion_window else 0
                avg_success = np.mean(success_window) if success_window else 0
                
                print(f"Episode {episode_count}, Step {total_training_steps}: "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Completion: {avg_completion:.2f}, "
                      f"Avg Success: {avg_success:.2f}")
            
            # Reset for new episode
            state, _ = train_env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_count += 1
    
    # Save final model
    agent.save(log_dir / f"sac_model_maps_{num_train_maps}_final.pt")
    print(f"Training on {num_train_maps} maps completed!")
    
    # Final evaluation
    train_reward, train_completion, train_success = evaluate_policy(agent, eval_train_env, num_episodes=20)
    val_reward, val_completion, val_success = evaluate_policy(agent, validation_env, num_episodes=20)
    
    print("\nFinal Evaluation Results:")
    print(f"  Training Env: Reward: {train_reward:.2f}, Completion: {train_completion:.2f}, Success: {train_success:.2f}")
    print(f"  Validation Env: Reward: {val_reward:.2f}, Completion: {val_completion:.2f}, Success: {val_success:.2f}")
    
    # Save final evaluation results
    final_results = {
        "num_maps": num_train_maps,
        "train_reward": train_reward,
        "train_completion": train_completion,
        "train_success": train_success,
        "val_reward": val_reward,
        "val_completion": val_completion,
        "val_success": val_success,
    }
    
    with open(log_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
    
    # Clean up
    train_env.close()
    eval_train_env.close()
    validation_env.close()
    
    # Return final evaluation metrics
    return train_completion, train_success, val_completion, val_success


def run_generalization_experiment():
    """Run experiments for different numbers of training maps."""
    # List of number of maps to test
    map_counts = [1, 3, 10]
    
    # Results storage
    results = {
        "map_counts": map_counts,
        "train_completions": [],
        "train_success_rates": [],
        "val_completions": [],
        "val_success_rates": []
    }
    
    # Run experiment for each map count
    for num_maps in map_counts:
        print(f"\n{'='*50}")
        print(f"Starting experiment with {num_maps} training maps")
        print(f"{'='*50}\n")
        
        train_completion, train_success, val_completion, val_success = train_and_evaluate(
            num_train_maps=num_maps,
            eval_interval=20000,  # Evaluate less frequently to speed up training
            total_steps=300000    # Reduced steps for faster experiment completion
        )
        
        # Store results
        results["train_completions"].append(train_completion)
        results["train_success_rates"].append(train_success)
        results["val_completions"].append(val_completion)
        results["val_success_rates"].append(val_success)
    
    # Save combined results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = FOLDER_ROOT / f"generalization_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(results_dir / "generalization_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    plot_generalization_curves(results, results_dir)


def plot_generalization_curves(results, save_dir):
    """Plot the generalization curves for route completion and success rate."""
    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 10))
    
    # Plot route completion
    plt.subplot(2, 1, 1)
    plt.plot(results["map_counts"], results["train_completions"], 'b-o', linewidth=2, markersize=8, label='Training')
    plt.plot(results["map_counts"], results["val_completions"], 'r-o', linewidth=2, markersize=8, label='Validation')
    plt.xlabel('Number of Training Maps', fontsize=14)
    plt.ylabel('Route Completion', fontsize=14)
    plt.title('Generalization: Route Completion', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(results["map_counts"])
    plt.ylim(0, 1.0)
    
    # Plot success rate
    plt.subplot(2, 1, 2)
    plt.plot(results["map_counts"], results["train_success_rates"], 'b-o', linewidth=2, markersize=8, label='Training')
    plt.plot(results["map_counts"], results["val_success_rates"], 'r-o', linewidth=2, markersize=8, label='Validation')
    plt.xlabel('Number of Training Maps', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.title('Generalization: Success Rate', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(results["map_counts"])
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_dir / "generalization_curves.png", dpi=300)
    plt.savefig(save_dir / "generalization_curves.pdf")
    
    print(f"Generalization curves saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run generalization experiments for SAC on MetaDrive")
    args = parser.parse_args()
    
    run_generalization_experiment()