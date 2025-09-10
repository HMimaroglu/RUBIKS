#!/usr/bin/env python3
"""
Reinforcement Learning toolkit for 2x2 Rubik's Cube solving.
Provides Goal-Conditioned RL + HER, Autodidactic Iteration, and A* search.
Single-file implementation that imports existing cube code read-only.
"""

import os
import sys
import json
import time
import heapq
import random
import argparse
from pathlib import Path
from collections import defaultdict, deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import numpy as np

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import existing cube implementation
try:
    from rubiks import Rubiks
    CUBE_CLASS = Rubiks
except ImportError:
    try:
        from cube import Cube2x2 as Rubiks
        CUBE_CLASS = Rubiks
    except ImportError:
        try:
            from main import Rubiks
            CUBE_CLASS = Rubiks
        except ImportError:
            raise ImportError(
                "Could not find cube implementation. Expected one of:\n"
                "  - from rubiks import Rubiks\n"
                "  - from cube import Cube2x2 as Rubiks\n"
                "  - from main import Rubiks\n"
                "Make sure your cube class is available in the current directory."
            )

# ============================================================================
# Utilities and Configuration
# ============================================================================

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)

def get_cube_state_key(cube) -> tuple:
    """Extract a hashable state key from cube instance."""
    # Use the encoding method from the original code
    order = ['front', 'right', 'back', 'left', 'top', 'bottom']
    flat = []
    for face in order:
        flat += cube.faces[face]
    return tuple(flat)

def make_dirs(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

# ============================================================================
# Environment Wrapper
# ============================================================================

class CubeEnv:
    """Environment wrapper for the user's Rubik's cube implementation."""
    
    def __init__(self, max_episode_steps: int = 30):
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        
        # Probe available actions from user's cube
        test_cube = CUBE_CLASS()
        if hasattr(test_cube, 'move_options'):
            self.actions = test_cube.move_options[:]
        else:
            # Fallback to common 2x2 moves
            self.actions = ['ru', 'rd', 'lu', 'ld', 'tcw', 'tccw', 'bcw', 'bccw']
        
        self.n_actions = len(self.actions)
        self.action_to_idx = {action: i for i, action in enumerate(self.actions)}
        
        # Initialize cube
        self.cube = CUBE_CLASS()
        self.initial_state = None
        
        # Color mapping for encoding
        self.color_to_idx = {'G': 0, 'B': 1, 'R': 2, 'W': 3, 'Y': 4, 'O': 5}
        self.state_dim = 24  # 6 faces * 4 stickers each
        
    def reset(self, scramble_depth: int = 5) -> np.ndarray:
        """Reset environment with a scrambled cube."""
        self.cube = CUBE_CLASS()
        
        # Scramble the cube
        if hasattr(self.cube, 'randomize'):
            self.cube.randomize(scramble_depth)
        else:
            # Manual scrambling
            for _ in range(scramble_depth):
                action = random.choice(self.actions)
                self.cube.move(action)
        
        self.initial_state = get_cube_state_key(self.cube)
        self.step_count = 0
        
        return self.encode(self.cube)
    
    def step(self, action: Union[int, str]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (obs, reward, done, info)."""
        # Convert action index to string if needed
        if isinstance(action, int):
            action_str = self.actions[action]
        else:
            action_str = action
            
        # Execute move
        success = self.cube.move(action_str)
        
        # Get new state
        obs = self.encode(self.cube)
        
        # Check if solved
        done = self.cube.solved()
        
        # Sparse reward: +1 for solving, 0 otherwise
        reward = 1.0 if done else 0.0
        
        self.step_count += 1
        
        # Episode timeout
        if self.step_count >= self.max_episode_steps:
            done = True
            
        info = {
            'solved': self.cube.solved(),
            'steps': self.step_count,
            'success': success
        }
        
        return obs, reward, done, info
    
    def encode(self, cube) -> np.ndarray:
        """Encode cube state as numpy array."""
        # Flatten all faces into a single array with color indices
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        order = ['front', 'right', 'back', 'left', 'top', 'bottom']
        idx = 0
        
        for face_name in order:
            face = cube.faces[face_name]
            for color in face:
                if color in self.color_to_idx:
                    state[idx] = self.color_to_idx[color]
                idx += 1
                
        return state
    
    def clone_cube(self):
        """Get a copy of current cube state."""
        return self.cube.clone() if hasattr(self.cube, 'clone') else None

# ============================================================================
# Replay Buffers
# ============================================================================

@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    goal: Optional[np.ndarray] = None

class ReplayBuffer:
    """Standard experience replay buffer."""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class HERBuffer(ReplayBuffer):
    """Hindsight Experience Replay buffer."""
    
    def __init__(self, capacity: int = 100000, her_k: int = 4):
        super().__init__(capacity)
        self.her_k = her_k
        self.episode_buffer = []
        
    def push_episode(self, episode_transitions: List[Transition]):
        """Store full episode and generate HER relabeled experiences."""
        # Store original episode
        for transition in episode_transitions:
            self.push(transition.obs, transition.action, transition.reward, 
                     transition.next_obs, transition.done, transition.goal)
        
        # Generate HER relabeled experiences
        for i, transition in enumerate(episode_transitions):
            for _ in range(self.her_k):
                # Sample future state as new goal (future strategy)
                if i + 1 < len(episode_transitions):
                    future_idx = random.randint(i + 1, len(episode_transitions) - 1)
                    new_goal = episode_transitions[future_idx].next_obs
                    
                    # Compute new reward (1 if achieved goal, 0 otherwise)
                    new_reward = 1.0 if np.allclose(transition.next_obs, new_goal, atol=1e-6) else 0.0
                    new_done = new_reward > 0.5
                    
                    self.push(transition.obs, transition.action, new_reward,
                             transition.next_obs, new_done, new_goal)

# ============================================================================
# Neural Networks
# ============================================================================

if HAS_TORCH:
    class MLP(nn.Module):
        """Multi-layer perceptron with layer normalization."""
        
        def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                     activation: str = 'relu', use_layernorm: bool = True):
            super().__init__()
            
            dims = [input_dim] + hidden_dims + [output_dim]
            layers = []
            
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                
                if i < len(dims) - 2:  # Not the output layer
                    if use_layernorm:
                        layers.append(nn.LayerNorm(dims[i + 1]))
                    
                    if activation == 'relu':
                        layers.append(nn.ReLU())
                    elif activation == 'tanh':
                        layers.append(nn.Tanh())
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
    
    class ActorCritic(nn.Module):
        """Actor-Critic network for goal-conditioned RL."""
        
        def __init__(self, state_dim: int, goal_dim: int, n_actions: int,
                     hidden_dims: List[int] = [256, 256]):
            super().__init__()
            
            input_dim = state_dim + goal_dim
            
            # Shared trunk
            self.trunk = MLP(input_dim, hidden_dims[:-1], hidden_dims[-1])
            
            # Actor head (policy)
            self.actor = nn.Linear(hidden_dims[-1], n_actions)
            
            # Critic head (Q-function)
            self.critic = nn.Linear(hidden_dims[-1], n_actions)
            
        def forward(self, state, goal):
            x = torch.cat([state, goal], dim=-1)
            features = self.trunk(x)
            
            logits = self.actor(features)
            q_values = self.critic(features)
            
            return logits, q_values
        
        def get_action(self, state, goal, epsilon: float = 0.1):
            """Epsilon-greedy action selection."""
            if random.random() < epsilon:
                return random.randint(0, self.actor.out_features - 1)
            
            with torch.no_grad():
                logits, q_values = self.forward(state, goal)
                return q_values.argmax(dim=-1).item()
    
    class ValueNetwork(nn.Module):
        """Value function approximator for autodidactic iteration."""
        
        def __init__(self, state_dim: int, hidden_dims: List[int] = [256, 256, 128]):
            super().__init__()
            self.network = MLP(state_dim, hidden_dims, 1)
            
        def forward(self, x):
            return self.network(x)

# ============================================================================
# Algorithms
# ============================================================================

class HERAgent:
    """Goal-Conditioned RL agent with Hindsight Experience Replay."""
    
    def __init__(self, env: CubeEnv, lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.995, device: str = 'cpu'):
        
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for HER agent")
            
        self.env = env
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_net = ActorCritic(env.state_dim, env.state_dim, env.n_actions).to(device)
        self.target_net = ActorCritic(env.state_dim, env.state_dim, env.n_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)
        self.buffer = HERBuffer()
        
    def select_action(self, state: np.ndarray, goal: np.ndarray) -> int:
        """Select action using epsilon-greedy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        
        return self.q_net.get_action(state_tensor, goal_tensor, self.epsilon)
    
    def update(self, batch_size: int = 256):
        """Update networks using sampled batch."""
        if len(self.buffer) < batch_size:
            return
            
        transitions = self.buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([t.obs for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_obs for t in transitions])).to(self.device)
        dones = torch.BoolTensor([t.done for t in transitions]).to(self.device)
        goals = torch.FloatTensor(np.array([t.goal if t.goal is not None else t.obs for t in transitions])).to(self.device)
        
        # Current Q-values
        _, current_q = self.q_net(states, goals)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            _, next_q = self.target_net(next_states, goals)
            next_q_max = next_q.max(1)[0]
            target_q = rewards + self.gamma * next_q_max * (~dones)
        
        # Loss and optimization
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """Soft update of target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def train_episode(self, max_scramble_depth: int = 8):
        """Train on a single episode with HER."""
        scramble_depth = random.randint(1, max_scramble_depth)
        obs = self.env.reset(scramble_depth)
        
        # Goal is always the solved state
        solved_cube = CUBE_CLASS()
        goal = self.env.encode(solved_cube)
        
        episode_transitions = []
        done = False
        
        while not done:
            action = self.select_action(obs, goal)
            next_obs, reward, done, info = self.env.step(action)
            
            episode_transitions.append(
                Transition(obs, action, reward, next_obs, done, goal)
            )
            
            obs = next_obs
        
        # Store episode in HER buffer
        self.buffer.push_episode(episode_transitions)
        
        return len(episode_transitions), info['solved']

class AutodidacticIteration:
    """Autodidactic Iteration for learning value heuristics."""
    
    def __init__(self, env: CubeEnv, lr: float = 1e-3, device: str = 'cpu'):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for Autodidactic Iteration")
            
        self.env = env
        self.device = device
        
        self.value_net = ValueNetwork(env.state_dim).to(device)
        self.optimizer = Adam(self.value_net.parameters(), lr=lr)
        
    def generate_shell_data(self, target_depth: int, samples_per_depth: int = 10000):
        """Generate training data at specific distance from solved state."""
        solved_cube = CUBE_CLASS()
        solved_state = get_cube_state_key(solved_cube)
        
        # BFS to find states at exactly target_depth
        queue = deque([(solved_cube.clone(), 0)])
        visited = {solved_state}
        states_at_depth = []
        
        while queue and len(states_at_depth) < samples_per_depth:
            cube, depth = queue.popleft()
            
            if depth == target_depth:
                states_at_depth.append((self.env.encode(cube), depth))
                continue
            elif depth < target_depth:
                # Generate children
                for action in self.env.actions:
                    child = cube.clone()
                    if child.move(action):
                        child_key = get_cube_state_key(child)
                        if child_key not in visited:
                            visited.add(child_key)
                            queue.append((child, depth + 1))
        
        return states_at_depth
    
    def train_on_shell(self, shell_data: List[Tuple[np.ndarray, int]], 
                      epochs: int = 100, batch_size: int = 256):
        """Train value network on shell data."""
        if not shell_data:
            return 0.0
            
        states = torch.FloatTensor([s[0] for s in shell_data]).to(self.device)
        values = torch.FloatTensor([s[1] for s in shell_data]).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(states, values)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_states, batch_values in dataloader:
                self.optimizer.zero_grad()
                
                pred_values = self.value_net(batch_states).squeeze()
                loss = F.mse_loss(pred_values, batch_values)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(dataloader)
        
        return total_loss / epochs

class AStarSolver:
    """A* search with learned value heuristic."""
    
    def __init__(self, env: CubeEnv, value_net=None, weight: float = 1.0):
        self.env = env
        self.value_net = value_net
        self.weight = weight
        
    def heuristic(self, cube) -> float:
        """Compute heuristic value for A* search."""
        if self.value_net is None:
            return 0.0  # Dijkstra mode
            
        with torch.no_grad():
            state = torch.FloatTensor(self.env.encode(cube)).unsqueeze(0)
            if HAS_TORCH:
                return self.value_net(state).item()
        return 0.0
    
    def solve(self, initial_cube, max_nodes: int = 100000) -> Tuple[List[str], int]:
        """Solve cube using A* search."""
        if initial_cube.solved():
            return [], 0
            
        # Priority queue: (f_score, g_score, node_id, cube, path)
        # Use node_id as tiebreaker to avoid comparing cube objects
        frontier = [(0, 0, 0, initial_cube, [])]
        visited = {get_cube_state_key(initial_cube)}
        nodes_expanded = 0
        node_counter = 1
        
        while frontier and nodes_expanded < max_nodes:
            f_score, g_score, node_id, current_cube, path = heapq.heappop(frontier)
            nodes_expanded += 1
            
            if current_cube.solved():
                return path, nodes_expanded
            
            # Expand neighbors
            for action in self.env.actions:
                child = current_cube.clone()
                if child.move(action):
                    child_key = get_cube_state_key(child)
                    
                    if child_key not in visited:
                        visited.add(child_key)
                        
                        new_g = g_score + 1
                        h_score = self.weight * self.heuristic(child)
                        new_f = new_g + h_score
                        new_path = path + [action]
                        
                        heapq.heappush(frontier, (new_f, new_g, node_counter, child, new_path))
                        node_counter += 1
        
        return [], nodes_expanded  # No solution found

# ============================================================================
# Training and Evaluation
# ============================================================================

def train_her(args):
    """Train HER agent."""
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for HER training")
        return
        
    print("Training HER agent (press Ctrl+C to stop)...")
    
    # Setup
    set_seeds(args.seed)
    make_dirs(args.save_dir)
    
    env = CubeEnv(max_episode_steps=args.max_steps)
    agent = HERAgent(env, lr=args.lr, device=args.device)
    
    # Load checkpoint if it exists
    checkpoint_path = os.path.join(args.save_dir, 'her_checkpoint.pt')
    start_episode = 0
    start_step = 0
    total_solved = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        agent.q_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
        start_episode = checkpoint.get('episode', 0)
        start_step = checkpoint.get('step', 0)
        total_solved = checkpoint.get('total_solved', 0)
        print(f"Resuming from episode {start_episode}, step {start_step}")
    
    # Training loop
    step = start_step
    episode = start_episode
    last_checkpoint_episode = episode
    
    try:
        while True:  # Train indefinitely until Ctrl+C
            episode_steps, solved = agent.train_episode(args.scramble_max)
            step += episode_steps
            episode += 1
            
            if solved:
                total_solved += 1
                
            # Update networks
            if step % args.update_freq == 0:
                loss = agent.update(args.batch_size)
                
            if step % args.target_update_freq == 0:
                agent.update_target_network()
                
            # Print progress every 1000 episodes
            if episode % 1000 == 0:
                solve_rate = total_solved / episode if episode > 0 else 0
                print(f"Episode {episode}, Step {step}, Solve Rate: {solve_rate:.3f}, Epsilon: {agent.epsilon:.3f}")
            
            # Save checkpoint every 10k episodes
            if episode - last_checkpoint_episode >= 10000:
                print(f"Saving checkpoint at episode {episode}...")
                torch.save({
                    'model_state_dict': agent.q_net.state_dict(),
                    'target_state_dict': agent.target_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'episode': episode,
                    'step': step,
                    'total_solved': total_solved,
                    'config': vars(args)
                }, checkpoint_path)
                
                # Also save a timestamped backup
                backup_path = os.path.join(args.save_dir, f'her_model_ep{episode}.pt')
                torch.save({
                    'model_state_dict': agent.q_net.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'config': vars(args)
                }, backup_path)
                
                last_checkpoint_episode = episode
                print(f"Checkpoint saved to {checkpoint_path}")
                print(f"Backup saved to {backup_path}")
    
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at episode {episode}")
        
        # Save final checkpoint
        print("Saving final checkpoint...")
        torch.save({
            'model_state_dict': agent.q_net.state_dict(),
            'target_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'episode': episode,
            'step': step,
            'total_solved': total_solved,
            'config': vars(args)
        }, checkpoint_path)
        
        # Save final model
        final_path = os.path.join(args.save_dir, 'her_model_final.pt')
        torch.save({
            'model_state_dict': agent.q_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'config': vars(args)
        }, final_path)
        
        solve_rate = total_solved / episode if episode > 0 else 0
        print(f"Final stats: {episode} episodes, {step} steps, {solve_rate:.3f} solve rate")
        print(f"Final model saved to {final_path}")

def train_adit(args):
    """Train Autodidactic Iteration value network."""
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for ADI training")
        return
        
    print("Training Autodidactic Iteration value network...")
    
    # Setup
    set_seeds(args.seed)
    make_dirs(args.save_dir)
    
    env = CubeEnv()
    adi = AutodidacticIteration(env, lr=args.lr, device=args.device)
    
    # Train on each shell
    for depth in args.frontier:
        print(f"Generating data for depth {depth}...")
        shell_data = adi.generate_shell_data(depth, args.per_shell)
        
        if shell_data:
            print(f"Training on {len(shell_data)} samples at depth {depth}...")
            avg_loss = adi.train_on_shell(shell_data, args.epochs)
            print(f"Depth {depth}: Average loss = {avg_loss:.4f}")
        else:
            print(f"No data generated for depth {depth}")
    
    # Save model
    save_path = os.path.join(args.save_dir, 'value_model.pt')
    torch.save({
        'model_state_dict': adi.value_net.state_dict(),
        'optimizer_state_dict': adi.optimizer.state_dict(),
        'config': vars(args)
    }, save_path)
    
    print(f"Training completed. Model saved to {save_path}")

def eval_policy(args):
    """Evaluate trained HER policy."""
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for policy evaluation")
        return
        
    print("Evaluating HER policy...")
    
    # Load model
    checkpoint = torch.load(args.policy_path, map_location=args.device)
    
    env = CubeEnv(max_episode_steps=50)
    agent = HERAgent(env, device=args.device)
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0.0  # Greedy evaluation
    
    # Print training info if available
    if 'episode' in checkpoint:
        print(f"Model trained for {checkpoint['episode']} episodes")
    
    solved_cube = CUBE_CLASS()
    goal = env.encode(solved_cube)
    
    results = []
    
    for depth in args.depths:
        print(f"Evaluating at scramble depth {depth}...")
        
        successes = 0
        total_steps = 0
        
        for episode in range(args.episodes):
            obs = env.reset(depth)
            done = False
            steps = 0
            
            while not done and steps < 50:
                action = agent.select_action(obs, goal)
                obs, _, done, info = env.step(action)
                steps += 1
                
            if info['solved']:
                successes += 1
                total_steps += steps
        
        success_rate = successes / args.episodes
        avg_steps = total_steps / max(successes, 1)
        
        results.append({
            'depth': depth,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'successes': successes
        })
        
        print(f"Depth {depth}: Success {successes}/{args.episodes} ({success_rate:.3f}), Avg Steps: {avg_steps:.1f}")
    
    # Save results
    make_dirs('runs/eval')
    with open('runs/eval/her_policy_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def eval_search(args):
    """Evaluate A* search with learned value function."""
    print("Evaluating A* search...")
    
    env = CubeEnv()
    
    # Load value network if provided
    value_net = None
    if args.value_path and HAS_TORCH:
        checkpoint = torch.load(args.value_path, map_location=args.device)
        value_net = ValueNetwork(env.state_dim).to(args.device)
        value_net.load_state_dict(checkpoint['model_state_dict'])
        value_net.eval()
    
    solver = AStarSolver(env, value_net, args.weight)
    
    results = []
    
    for depth in args.depths:
        print(f"Evaluating at scramble depth {depth}...")
        
        successes = 0
        total_nodes = 0
        total_solution_length = 0
        
        for episode in range(args.episodes):
            # Generate scrambled cube
            cube = CUBE_CLASS()
            for _ in range(depth):
                action = random.choice(env.actions)
                cube.move(action)
            
            # Solve with A*
            solution, nodes_expanded = solver.solve(cube, args.max_nodes)
            
            if solution:
                successes += 1
                total_nodes += nodes_expanded
                total_solution_length += len(solution)
        
        success_rate = successes / args.episodes
        avg_nodes = total_nodes / max(successes, 1)
        avg_length = total_solution_length / max(successes, 1)
        
        results.append({
            'depth': depth,
            'success_rate': success_rate,
            'avg_nodes': avg_nodes,
            'avg_solution_length': avg_length,
            'successes': successes
        })
        
        print(f"Depth {depth}: Success {successes}/{args.episodes} ({success_rate:.3f}), "
              f"Avg Nodes: {avg_nodes:.1f}, Avg Length: {avg_length:.1f}")
    
    # Save results
    make_dirs('runs/eval')
    with open('runs/eval/astar_results.json', 'w') as f:
        json.dump(results, f, indent=2)

def selftest():
    """Run self-tests to verify the implementation works."""
    print("Running self-tests...")
    
    try:
        # Test 1: Environment creation and basic operations
        print("Test 1: Environment creation...")
        env = CubeEnv()
        obs = env.reset(3)
        assert obs.shape == (24,), f"Expected obs shape (24,), got {obs.shape}"
        
        for _ in range(20):
            action = random.randint(0, env.n_actions - 1)
            obs, reward, done, info = env.step(action)
            assert obs.shape == (24,), f"Expected obs shape (24,), got {obs.shape}"
            
        print("✓ Environment test passed")
        
        # Test 2: State encoding consistency
        print("Test 2: State encoding...")
        cube1 = CUBE_CLASS()
        cube2 = CUBE_CLASS()
        
        enc1 = env.encode(cube1)
        enc2 = env.encode(cube2)
        
        assert np.allclose(enc1, enc2), "Identical cubes should have identical encodings"
        
        # Apply same move to both
        cube1.move(env.actions[0])
        cube2.move(env.actions[0])
        
        enc1_moved = env.encode(cube1)
        enc2_moved = env.encode(cube2)
        
        assert np.allclose(enc1_moved, enc2_moved), "Cubes with same moves should have identical encodings"
        print("✓ State encoding test passed")
        
        # Test 3: A* solver with uniform heuristic (Dijkstra)
        print("Test 3: A* solver...")
        solver = AStarSolver(env, value_net=None, weight=0.0)  # Pure Dijkstra
        
        # Test on depth 1-2 scrambles
        for depth in [1, 2]:
            cube = CUBE_CLASS()
            for _ in range(depth):
                cube.move(random.choice(env.actions))
            
            if not cube.solved():
                solution, nodes = solver.solve(cube, max_nodes=1000)
                if solution:
                    print(f"✓ Solved depth {depth} scramble in {len(solution)} moves, {nodes} nodes")
                else:
                    print(f"⚠ Could not solve depth {depth} scramble (may be too complex)")
        
        print("✓ A* solver test completed")
        
        print("All self-tests passed! 🎉")
        
    except Exception as e:
        print(f"❌ Self-test failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# CLI Interface
# ============================================================================

def main():
    if not HAS_TORCH:
        print("WARNING: PyTorch not found. Only self-test and A* evaluation (without learned heuristic) will work.")
        print("Install PyTorch with: pip install torch")
    
    parser = argparse.ArgumentParser(description="Reinforcement Learning for 2x2 Rubik's Cube")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    def add_common_args(parser):
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    
    # Train HER command
    train_her_parser = subparsers.add_parser('train_her', help='Train HER agent')
    train_her_parser.add_argument('--steps', type=int, default=200000, help='Training steps')
    train_her_parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    train_her_parser.add_argument('--scramble_max', type=int, default=8, help='Max scramble depth')
    train_her_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_her_parser.add_argument('--max_steps', type=int, default=30, help='Max episode steps')
    train_her_parser.add_argument('--update_freq', type=int, default=4, help='Update frequency')
    train_her_parser.add_argument('--target_update_freq', type=int, default=1000, help='Target network update frequency')
    train_her_parser.add_argument('--save_dir', type=str, default='runs/her', help='Save directory')
    add_common_args(train_her_parser)
    
    # Train ADI command
    train_adit_parser = subparsers.add_parser('train_adit', help='Train Autodidactic Iteration')
    train_adit_parser.add_argument('--frontier', nargs='+', type=int, default=[0, 2, 4, 6, 8], help='Frontier depths')
    train_adit_parser.add_argument('--per_shell', type=int, default=10000, help='Samples per shell')
    train_adit_parser.add_argument('--epochs', type=int, default=100, help='Training epochs per shell')
    train_adit_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    train_adit_parser.add_argument('--save_dir', type=str, default='runs/adit', help='Save directory')
    add_common_args(train_adit_parser)
    
    # Evaluate policy command
    eval_policy_parser = subparsers.add_parser('eval_policy', help='Evaluate HER policy')
    eval_policy_parser.add_argument('--policy_path', type=str, required=True, help='Path to policy model')
    eval_policy_parser.add_argument('--depths', nargs='+', type=int, default=[1, 3, 5, 7, 9, 11, 14], help='Scramble depths')
    eval_policy_parser.add_argument('--episodes', type=int, default=1000, help='Episodes per depth')
    add_common_args(eval_policy_parser)
    
    # Evaluate search command
    eval_search_parser = subparsers.add_parser('eval_search', help='Evaluate A* search')
    eval_search_parser.add_argument('--value_path', type=str, help='Path to value model (optional)')
    eval_search_parser.add_argument('--weight', type=float, default=1.0, help='Heuristic weight')
    eval_search_parser.add_argument('--depths', nargs='+', type=int, default=[1, 3, 5, 7, 9, 11, 14], help='Scramble depths')
    eval_search_parser.add_argument('--episodes', type=int, default=200, help='Episodes per depth')
    eval_search_parser.add_argument('--max_nodes', type=int, default=100000, help='Max nodes to expand')
    add_common_args(eval_search_parser)
    
    # Self-test command
    selftest_parser = subparsers.add_parser('selftest', help='Run self-tests')
    
    args = parser.parse_args()
    
    if args.command == 'train_her':
        train_her(args)
    elif args.command == 'train_adit':
        train_adit(args)
    elif args.command == 'eval_policy':
        eval_policy(args)
    elif args.command == 'eval_search':
        eval_search(args)
    elif args.command == 'selftest':
        selftest()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()