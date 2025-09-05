import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from rubiks import Rubiks
from rl import RubiksSolver, DQN

def load_checkpoint_data(checkpoint_path):
    """Load checkpoint and extract key metrics"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return {
        'epoch': checkpoint['epoch'],
        'epsilon': checkpoint['epsilon'],
        'memory_size': len(checkpoint['memory']),
        'model_state': checkpoint['model_state_dict']
    }

def test_model_performance(agent, num_tests=100, max_scrambles=10):
    """Test model performance on scrambled cubes"""
    results = {
        'solved': 0,
        'total_rewards': [],
        'solve_steps': [],
        'failed_attempts': 0
    }
    
    for test in range(num_tests):
        # Create and scramble cube
        cube = Rubiks()
        scramble_moves = np.random.randint(1, max_scrambles + 1)
        for _ in range(scramble_moves):
            move = np.random.choice(cube.move_options)
            cube.move(move)
        
        # Test if already solved (rare but possible)
        if cube.solved():
            continue
            
        # Try to solve with agent
        state = agent.state_to_tensor(cube)
        total_reward = 0
        max_steps = 200
        
        for step in range(max_steps):
            # Use trained policy (no exploration)
            with torch.no_grad():
                q_values = agent.q_network(state.unsqueeze(0))
                action = q_values.argmax().item()
            
            move = cube.move_options[action]
            cube.move(move)
            
            if cube.solved():
                results['solved'] += 1
                results['solve_steps'].append(step + 1)
                total_reward = 100 - (step + 1)  # Same reward structure as training
                results['total_rewards'].append(total_reward)
                break
                
            state = agent.state_to_tensor(cube)
        else:
            # Failed to solve within max_steps
            results['failed_attempts'] += 1
            results['total_rewards'].append(-max_steps)  # Penalty for failure
    
    return results

def analyze_all_checkpoints():
    """Analyze all available checkpoints"""
    checkpoint_dir = 'checkpoint'
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found!")
        return
    
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    print("Analyzing checkpoints...\n")
    
    analysis_results = {}
    
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        # Load checkpoint data
        checkpoint_data = load_checkpoint_data(checkpoint_path)
        epoch = checkpoint_data['epoch']
        
        print(f"Analyzing Epoch {epoch}...")
        
        # Create agent and load model
        agent = RubiksSolver()
        agent.q_network.load_state_dict(checkpoint_data['model_state'])
        agent.q_network.eval()  # Set to evaluation mode
        agent.epsilon = 0  # No exploration during testing
        
        # Test performance
        test_results = test_model_performance(agent, num_tests=50)
        
        # Store results
        analysis_results[epoch] = {
            'checkpoint_data': checkpoint_data,
            'test_results': test_results,
            'solve_rate': test_results['solved'] / 50 * 100,
            'avg_reward': np.mean(test_results['total_rewards']) if test_results['total_rewards'] else -200,
            'avg_solve_steps': np.mean(test_results['solve_steps']) if test_results['solve_steps'] else None
        }
        
        print(f"  Solve Rate: {analysis_results[epoch]['solve_rate']:.1f}%")
        print(f"  Avg Reward: {analysis_results[epoch]['avg_reward']:.2f}")
        if analysis_results[epoch]['avg_solve_steps']:
            print(f"  Avg Steps to Solve: {analysis_results[epoch]['avg_solve_steps']:.1f}")
        print(f"  Epsilon: {checkpoint_data['epsilon']:.4f}")
        print()
    
    return analysis_results

def plot_training_progress(analysis_results):
    """Create visualization of training progress"""
    epochs = sorted(analysis_results.keys())
    solve_rates = [analysis_results[epoch]['solve_rate'] for epoch in epochs]
    avg_rewards = [analysis_results[epoch]['avg_reward'] for epoch in epochs]
    epsilons = [analysis_results[epoch]['checkpoint_data']['epsilon'] for epoch in epochs]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Solve rate over time
    ax1.plot(epochs, solve_rates, 'b-o', linewidth=2, markersize=6)
    ax1.set_title('Solve Rate Over Training', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('Solve Rate (%)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Average reward over time
    ax2.plot(epochs, avg_rewards, 'g-o', linewidth=2, markersize=6)
    ax2.set_title('Average Reward Over Training', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax2.legend()
    
    # Epsilon decay over time
    ax3.plot(epochs, epsilons, 'r-o', linewidth=2, markersize=6)
    ax3.set_title('Exploration (Epsilon) Over Training', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Training Epoch')
    ax3.set_ylabel('Epsilon Value')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    print("Training progress plot saved as 'training_progress.png'")
    plt.show()

def generate_report(analysis_results):
    """Generate detailed analysis report"""
    print("="*60)
    print("RUBIK'S CUBE RL TRAINING ANALYSIS REPORT")
    print("="*60)
    
    epochs = sorted(analysis_results.keys())
    
    # Overall progress
    initial_performance = analysis_results[epochs[0]]
    final_performance = analysis_results[epochs[-1]]
    
    print(f"\nTRAINING SUMMARY:")
    print(f"  Training Epochs: {epochs[0]} to {epochs[-1]} ({len(epochs)} checkpoints)")
    
    print(f"\nINITIAL PERFORMANCE (Epoch {epochs[0]}):")
    print(f"  Solve Rate: {initial_performance['solve_rate']:.1f}%")
    print(f"  Average Reward: {initial_performance['avg_reward']:.2f}")
    print(f"  Epsilon: {initial_performance['checkpoint_data']['epsilon']:.4f}")
    
    print(f"\nFINAL PERFORMANCE (Epoch {epochs[-1]}):")
    print(f"  Solve Rate: {final_performance['solve_rate']:.1f}%")
    print(f"  Average Reward: {final_performance['avg_reward']:.2f}")
    print(f"  Epsilon: {final_performance['checkpoint_data']['epsilon']:.4f}")
    
    # Improvement metrics
    solve_rate_improvement = final_performance['solve_rate'] - initial_performance['solve_rate']
    reward_improvement = final_performance['avg_reward'] - initial_performance['avg_reward']
    
    print(f"\nIMPROVEMENT:")
    print(f"  Solve Rate: {solve_rate_improvement:+.1f}% points")
    print(f"  Average Reward: {reward_improvement:+.2f} points")
    
    # Best performing epoch
    best_epoch = max(epochs, key=lambda e: analysis_results[e]['solve_rate'])
    best_performance = analysis_results[best_epoch]
    
    print(f"\nBEST PERFORMANCE (Epoch {best_epoch}):")
    print(f"  Solve Rate: {best_performance['solve_rate']:.1f}%")
    print(f"  Average Reward: {best_performance['avg_reward']:.2f}")
    if best_performance['avg_solve_steps']:
        print(f"  Average Steps to Solve: {best_performance['avg_solve_steps']:.1f}")
    
    print("\n" + "="*60)

def main():
    print("Rubik's Cube RL Model Analysis")
    print("="*40)
    
    # Analyze all checkpoints
    results = analyze_all_checkpoints()
    
    if not results:
        print("No analysis results to display.")
        return
    
    # Generate visualizations
    try:
        plot_training_progress(results)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Generate report
    generate_report(results)

if __name__ == "__main__":
    main()