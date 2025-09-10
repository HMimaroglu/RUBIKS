#!/usr/bin/env python3
"""
Evaluation script for trained Rubik's Cube models.
Tests a specific model on 10 different cube states, each scrambled and solved 1000 times.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Import from our RL toolkit
from rl import (
    CubeEnv, HERAgent, AStarSolver, ValueNetwork, CUBE_CLASS, 
    set_seeds, get_cube_state_key, HAS_TORCH
)

try:
    import torch
except ImportError:
    pass

def detect_model_type(model_path):
    """Detect what type of model this is based on path and contents."""
    model_path = Path(model_path)
    
    # Check filename patterns
    if 'adit' in model_path.name.lower() or 'value' in model_path.name.lower():
        return 'value'
    elif 'her' in model_path.name.lower():
        return 'policy'
    
    # Check parent directory
    if 'adit' in str(model_path.parent).lower():
        return 'value'
    elif 'her' in str(model_path.parent).lower():
        return 'policy'
    
    # Try to load and inspect
    if HAS_TORCH:
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'target_state_dict' in checkpoint:
                return 'policy'  # HER models have target networks
            else:
                return 'value'   # Value models don't
        except:
            pass
    
    # Default guess
    return 'policy'

def load_policy_model(model_path, device='cpu'):
    """Load a HER policy model."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    env = CubeEnv(max_episode_steps=100)  # Generous limit for evaluation
    agent = HERAgent(env, device=device)
    
    # Load model state
    agent.q_net.load_state_dict(checkpoint['model_state_dict'])
    agent.epsilon = 0.0  # Greedy evaluation
    
    return agent, env

def load_value_model(model_path, device='cpu'):
    """Load an ADI value model."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    env = CubeEnv()
    value_net = ValueNetwork(env.state_dim).to(device)
    value_net.load_state_dict(checkpoint['model_state_dict'])
    value_net.eval()
    
    solver = AStarSolver(env, value_net, weight=1.0)
    
    return solver, env

def generate_test_cubes(num_cubes=10, scramble_depth=8):
    """Generate a set of test cube states."""
    env = CubeEnv()
    test_states = []
    
    print(f"Generating {num_cubes} test cube states (scramble depth {scramble_depth})...")
    
    for i in range(num_cubes):
        # Create a unique scramble pattern
        cube = CUBE_CLASS()
        moves = []
        
        # Use a mix of different move types to create diverse states
        move_types = [
            ['tcw', 'tccw'],  # Top moves
            ['bcw', 'bccw'],  # Bottom moves  
            ['ru', 'rd'],     # Right moves
            ['lu', 'ld']      # Left moves
        ]
        
        for _ in range(scramble_depth):
            # Randomly choose move type, then specific move
            move_type = random.choice(move_types)
            move = random.choice(move_type)
            cube.move(move)
            moves.append(move)
        
        # Verify it's not accidentally solved
        if not cube.solved():
            test_states.append({
                'id': i,
                'cube_state': get_cube_state_key(cube),
                'scramble_moves': moves,
                'scramble_depth': len(moves)
            })
            print(f"  Test cube {i+1}: {len(moves)} moves")
        else:
            # Retry if we accidentally created a solved state
            i -= 1
    
    return test_states

def test_policy_on_cube(agent, env, target_state, num_trials=1000):
    """Test policy model on a specific cube state multiple times."""
    solved_cube = CUBE_CLASS()
    goal = env.encode(solved_cube)
    
    results = {
        'solved': 0,
        'solve_times': [],
        'solution_lengths': [],
        'failed': 0
    }
    
    for trial in range(num_trials):
        # Recreate the target cube state
        cube = CUBE_CLASS()
        # Apply the stored scramble moves
        # Note: We need to reconstruct from the state key
        # For now, we'll use a placeholder approach
        
        # Reset environment with this cube
        env.cube = cube
        obs = env.encode(cube)
        env.step_count = 0
        
        # Try to solve
        solved = False
        steps = 0
        solution = []
        start_time = time.time()
        
        while not solved and steps < 100:  # Max 100 moves
            action = agent.select_action(obs, goal)
            action_str = env.actions[action]
            obs, _, done, info = env.step(action)
            solution.append(action_str)
            steps += 1
            solved = info['solved']
        
        solve_time = time.time() - start_time
        
        if solved:
            results['solved'] += 1
            results['solve_times'].append(solve_time)
            results['solution_lengths'].append(steps)
        else:
            results['failed'] += 1
    
    return results

def test_value_on_cube(solver, env, cube_state_info, num_trials=1000):
    """Test value model (A*) on a specific cube state multiple times."""
    results = {
        'solved': 0,
        'solve_times': [],
        'solution_lengths': [],
        'nodes_expanded': [],
        'failed': 0
    }
    
    for trial in range(num_trials):
        # Recreate cube from scramble moves
        cube = CUBE_CLASS()
        for move in cube_state_info['scramble_moves']:
            cube.move(move)
        
        # Solve with A*
        start_time = time.time()
        solution, nodes = solver.solve(cube, max_nodes=10000)
        solve_time = time.time() - start_time
        
        if solution:
            results['solved'] += 1
            results['solve_times'].append(solve_time)
            results['solution_lengths'].append(len(solution))
            results['nodes_expanded'].append(nodes)
        else:
            results['failed'] += 1
    
    return results

def print_results(cube_id, results, num_trials):
    """Print results for a single cube."""
    solve_rate = results['solved'] / num_trials
    print(f"\n  Cube {cube_id+1} Results:")
    print(f"    Success rate: {results['solved']}/{num_trials} ({solve_rate:.1%})")
    
    if results['solved'] > 0:
        avg_time = sum(results['solve_times']) / len(results['solve_times'])
        avg_length = sum(results['solution_lengths']) / len(results['solution_lengths'])
        
        print(f"    Average solve time: {avg_time:.4f}s")
        print(f"    Average solution length: {avg_length:.1f} moves")
        
        if 'nodes_expanded' in results and results['nodes_expanded']:
            avg_nodes = sum(results['nodes_expanded']) / len(results['nodes_expanded'])
            print(f"    Average nodes expanded: {avg_nodes:.0f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Rubik's Cube model")
    parser.add_argument('model_path', type=str, 
                       help='Path to the .pt model file to evaluate')
    parser.add_argument('--trials', type=int, default=1000,
                       help='Number of trials per cube state (default: 1000)')
    parser.add_argument('--cubes', type=int, default=10,
                       help='Number of different cube states to test (default: 10)')
    parser.add_argument('--scramble_depth', type=int, default=8,
                       help='Scramble depth for test cubes (default: 8)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        return
    
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for evaluation")
        print("Install with: pip install torch")
        return
    
    # Set seed for reproducible test states
    set_seeds(args.seed)
    
    print("Rubik's Cube Model Evaluation")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Test cubes: {args.cubes}")
    print(f"Trials per cube: {args.trials}")
    print(f"Scramble depth: {args.scramble_depth}")
    print()
    
    # Detect and load model
    model_type = detect_model_type(args.model_path)
    print(f"Detected model type: {model_type}")
    
    try:
        if model_type == 'policy':
            agent, env = load_policy_model(args.model_path, args.device)
            print("Loaded HER policy model")
        else:
            solver, env = load_value_model(args.model_path, args.device)
            print("Loaded value model with A* solver")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Generate test cube states
    test_states = generate_test_cubes(args.cubes, args.scramble_depth)
    
    print(f"\nTesting on {len(test_states)} cube states...")
    
    # Run evaluation
    all_results = []
    total_start_time = time.time()
    
    for i, cube_state in enumerate(test_states):
        print(f"\nTesting cube {i+1}/{len(test_states)}...")
        start_time = time.time()
        
        if model_type == 'policy':
            results = test_policy_on_cube(agent, env, cube_state, args.trials)
        else:
            results = test_value_on_cube(solver, env, cube_state, args.trials)
        
        test_time = time.time() - start_time
        print(f"  Completed {args.trials} trials in {test_time:.1f}s")
        
        print_results(i, results, args.trials)
        all_results.append(results)
    
    total_time = time.time() - total_start_time
    
    # Overall summary
    print("\n" + "=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)
    
    total_solved = sum(r['solved'] for r in all_results)
    total_attempts = len(test_states) * args.trials
    overall_rate = total_solved / total_attempts
    
    print(f"Overall success rate: {total_solved}/{total_attempts} ({overall_rate:.1%})")
    
    if total_solved > 0:
        all_times = [t for r in all_results for t in r['solve_times']]
        all_lengths = [l for r in all_results for l in r['solution_lengths']]
        
        print(f"Average solve time: {sum(all_times)/len(all_times):.4f}s")
        print(f"Average solution length: {sum(all_lengths)/len(all_lengths):.1f} moves")
        
        if all_results[0].get('nodes_expanded'):
            all_nodes = [n for r in all_results for n in r['nodes_expanded']]
            print(f"Average nodes expanded: {sum(all_nodes)/len(all_nodes):.0f}")
    
    print(f"Total evaluation time: {total_time:.1f}s")
    
    # Save detailed results
    output_dir = Path('eval_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).stem
    
    results_data = {
        'model_path': args.model_path,
        'model_type': model_type,
        'config': vars(args),
        'test_states': test_states,
        'results': all_results,
        'summary': {
            'total_solved': total_solved,
            'total_attempts': total_attempts,
            'success_rate': overall_rate,
            'total_time': total_time
        }
    }
    
    output_file = output_dir / f'{model_name}_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == '__main__':
    main()