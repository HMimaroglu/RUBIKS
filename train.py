#!/usr/bin/env python3
"""
Training script for Rubik's Cube RL models.
Supports 5 different model architectures/approaches.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

# Import from our RL toolkit
from rl import (
    CubeEnv, HERAgent, AutodidacticIteration, CUBE_CLASS, 
    set_seeds, make_dirs, HAS_TORCH
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
except ImportError:
    pass

def train_her_basic(args):
    """Train basic HER agent (Model 1)."""
    print("Training HER Basic (Model 1)...")
    
    env = CubeEnv(max_episode_steps=30)
    agent = HERAgent(env, lr=1e-3, epsilon_decay=0.995, device=args.device)
    
    save_dir = f"models/her_basic_{args.suffix}" if args.suffix else "models/her_basic"
    return train_her_common(agent, env, save_dir, args)

def train_her_large(args):
    """Train large HER agent with bigger networks (Model 2)."""
    print("Training HER Large (Model 2)...")
    
    env = CubeEnv(max_episode_steps=50)
    agent = HERAgent(env, lr=5e-4, epsilon_decay=0.999, device=args.device)
    
    # Modify network architecture to be larger
    if HAS_TORCH:
        from rl import ActorCritic
        agent.q_net = ActorCritic(env.state_dim, env.state_dim, env.n_actions, 
                                 hidden_dims=[512, 512, 256]).to(args.device)
        agent.target_net = ActorCritic(env.state_dim, env.state_dim, env.n_actions,
                                      hidden_dims=[512, 512, 256]).to(args.device)
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        agent.optimizer = Adam(agent.q_net.parameters(), lr=5e-4)
    
    save_dir = f"models/her_large_{args.suffix}" if args.suffix else "models/her_large"
    return train_her_common(agent, env, save_dir, args)

def train_her_deep(args):
    """Train deep HER agent with more layers (Model 3)."""
    print("Training HER Deep (Model 3)...")
    
    env = CubeEnv(max_episode_steps=40)
    agent = HERAgent(env, lr=1e-3, epsilon_decay=0.997, device=args.device)
    
    # Modify network architecture to be deeper
    if HAS_TORCH:
        from rl import ActorCritic
        agent.q_net = ActorCritic(env.state_dim, env.state_dim, env.n_actions,
                                 hidden_dims=[256, 256, 256, 128]).to(args.device)
        agent.target_net = ActorCritic(env.state_dim, env.state_dim, env.n_actions,
                                      hidden_dims=[256, 256, 256, 128]).to(args.device)
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        agent.optimizer = Adam(agent.q_net.parameters(), lr=1e-3)
    
    save_dir = f"models/her_deep_{args.suffix}" if args.suffix else "models/her_deep"
    return train_her_common(agent, env, save_dir, args)

def train_her_fast(args):
    """Train fast HER agent with frequent updates (Model 4)."""
    print("Training HER Fast (Model 4)...")
    
    env = CubeEnv(max_episode_steps=25)
    agent = HERAgent(env, lr=2e-3, epsilon_decay=0.99, device=args.device)
    
    save_dir = f"models/her_fast_{args.suffix}" if args.suffix else "models/her_fast"
    
    # Custom training with more frequent updates
    make_dirs(save_dir)
    
    # Load checkpoint if it exists
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
    start_episode = 0
    total_solved = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        agent.q_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
        start_episode = checkpoint.get('episode', 0)
        total_solved = checkpoint.get('total_solved', 0)
    
    episode = start_episode
    last_checkpoint = episode
    
    try:
        while True:
            episode_steps, solved = agent.train_episode(10)  # Easier scrambles
            episode += 1
            
            if solved:
                total_solved += 1
            
            # Update every 2 steps (more frequent)
            if episode % 2 == 0:
                agent.update(128)  # Smaller batch
                
            if episode % 500 == 0:  # More frequent target updates
                agent.update_target_network()
                
            if episode % 1000 == 0:
                solve_rate = total_solved / episode if episode > 0 else 0
                print(f"Episode {episode}, Solve Rate: {solve_rate:.3f}, Epsilon: {agent.epsilon:.3f}")
            
            if episode - last_checkpoint >= 10000:
                print(f"Saving checkpoint at episode {episode}...")
                torch.save({
                    'model_state_dict': agent.q_net.state_dict(),
                    'target_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'episode': episode,
                    'total_solved': total_solved,
                    'config': vars(args)
                }, checkpoint_path)
                last_checkpoint = episode
                
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at episode {episode}")
        torch.save({
            'model_state_dict': agent.q_net.state_dict(),
            'target_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'episode': episode,
            'total_solved': total_solved,
            'config': vars(args)
        }, checkpoint_path)
        
        final_path = os.path.join(save_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': agent.q_net.state_dict(),
            'config': vars(args)
        }, final_path)
        
        solve_rate = total_solved / episode if episode > 0 else 0
        print(f"Final: {episode} episodes, {solve_rate:.3f} solve rate")
        print(f"Model saved to {final_path}")

def train_adit_value(args):
    """Train Autodidactic Iteration value network (Model 5)."""
    print("Training Autodidactic Iteration Value (Model 5)...")
    
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for ADI training")
        return
    
    env = CubeEnv()
    adi = AutodidacticIteration(env, lr=1e-3, device=args.device)
    
    save_dir = f"models/adit_value_{args.suffix}" if args.suffix else "models/adit_value"
    make_dirs(save_dir)
    
    # Extended frontier for better coverage
    frontiers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    samples_per_shell = 5000
    
    print("Training value network on distance shells...")
    
    for depth in frontiers:
        print(f"Generating data for depth {depth}...")
        shell_data = adi.generate_shell_data(depth, samples_per_shell)
        
        if shell_data:
            print(f"Training on {len(shell_data)} samples at depth {depth}...")
            avg_loss = adi.train_on_shell(shell_data, epochs=50)
            print(f"Depth {depth}: Average loss = {avg_loss:.4f}")
            
            # Save intermediate model
            intermediate_path = os.path.join(save_dir, f'value_depth_{depth}.pt')
            torch.save({
                'model_state_dict': adi.value_net.state_dict(),
                'depth': depth,
                'config': vars(args)
            }, intermediate_path)
        else:
            print(f"No data generated for depth {depth}")
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': adi.value_net.state_dict(),
        'config': vars(args)
    }, final_path)
    
    print(f"Value network training completed. Model saved to {final_path}")

def train_her_common(agent, env, save_dir, args):
    """Common training loop for HER variants."""
    make_dirs(save_dir)
    
    # Load checkpoint if exists
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')
    start_episode = 0
    total_solved = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        agent.q_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
        start_episode = checkpoint.get('episode', 0)
        total_solved = checkpoint.get('total_solved', 0)
    
    episode = start_episode
    last_checkpoint = episode
    
    try:
        while True:
            episode_steps, solved = agent.train_episode(12)  # Max scramble depth
            episode += 1
            
            if solved:
                total_solved += 1
            
            # Standard update frequency
            if episode % 4 == 0:
                agent.update(256)
                
            if episode % 1000 == 0:
                agent.update_target_network()
                
            if episode % 1000 == 0:
                solve_rate = total_solved / episode if episode > 0 else 0
                print(f"Episode {episode}, Solve Rate: {solve_rate:.3f}, Epsilon: {agent.epsilon:.3f}")
            
            # Save every 10k episodes
            if episode - last_checkpoint >= 10000:
                print(f"Saving checkpoint at episode {episode}...")
                torch.save({
                    'model_state_dict': agent.q_net.state_dict(),
                    'target_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'episode': episode,
                    'total_solved': total_solved,
                    'config': vars(args)
                }, checkpoint_path)
                last_checkpoint = episode
                
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at episode {episode}")
        
        # Save final checkpoint
        torch.save({
            'model_state_dict': agent.q_net.state_dict(),
            'target_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'episode': episode,
            'total_solved': total_solved,
            'config': vars(args)
        }, checkpoint_path)
        
        # Save final model
        final_path = os.path.join(save_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': agent.q_net.state_dict(),
            'config': vars(args)
        }, final_path)
        
        solve_rate = total_solved / episode if episode > 0 else 0
        print(f"Final: {episode} episodes, {solve_rate:.3f} solve rate")
        print(f"Model saved to {final_path}")

def main():
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for training")
        print("Install with: pip install torch")
        return
    
    parser = argparse.ArgumentParser(description="Train Rubik's Cube RL Models")
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4, 5], required=True,
                       help='Model to train:\n'
                            '1: HER Basic (standard)\n'
                            '2: HER Large (bigger networks)\n'
                            '3: HER Deep (more layers)\n'
                            '4: HER Fast (frequent updates)\n'
                            '5: ADI Value (heuristic learning)')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix for model directory (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seeds(args.seed)
    
    print(f"Training Model {args.model} on device: {args.device}")
    print("Press Ctrl+C to stop training and save model")
    print("-" * 50)
    
    # Route to appropriate training function
    if args.model == 1:
        train_her_basic(args)
    elif args.model == 2:
        train_her_large(args)
    elif args.model == 3:
        train_her_deep(args)
    elif args.model == 4:
        train_her_fast(args)
    elif args.model == 5:
        train_adit_value(args)

if __name__ == '__main__':
    main()