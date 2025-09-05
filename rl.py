import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import math
import time
from collections import deque
from rubiks import Rubiks
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Installing tqdm for progress bars...")
    os.system("pip install tqdm")
    from tqdm import tqdm
    HAS_TQDM = True

# Create checkpoint directory
os.makedirs('checkpoint', exist_ok=True)

class DQN(nn.Module):
    def __init__(self, input_size=54, hidden_size=1024, output_size=8, dropout_rate=0.2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class RubiksSolver:
    def __init__(self, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50000, gamma=0.5)
        
        self.memory = deque(maxlen=20000)  # Increased memory
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 64  # Increased batch size
        self.gamma = 0.99  # Higher discount factor
        
        # Curriculum learning parameters
        self.min_scrambles = 1
        self.max_scrambles = 15
        self.current_episode = 0
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def state_to_tensor(self, cube):
        # Convert cube state to flat tensor
        state = []
        for face_name in ['front', 'back', 'bottom', 'right', 'left', 'top']:
            for color in cube.faces[face_name]:
                # Convert color letters to numbers
                color_map = {'G': 0, 'B': 1, 'R': 2, 'W': 3, 'Y': 4, 'O': 5}
                state.append(color_map[color])
        return torch.FloatTensor(state).to(self.device)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 7)
        
        q_values = self.q_network(state.unsqueeze(0))
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_curriculum_scrambles(self, episode):
        # Curriculum learning: gradually increase difficulty
        progress = min(episode / 500000, 1.0)  # Reach max difficulty by episode 500k
        current_max = self.min_scrambles + (self.max_scrambles - self.min_scrambles) * progress
        return random.randint(self.min_scrambles, int(current_max))
    
    def calculate_improved_reward(self, cube, step, solved, initial_scrambles):
        if solved:
            # Higher reward for solving with fewer moves
            efficiency_bonus = max(0, 50 - step)
            difficulty_bonus = initial_scrambles * 5  # Bonus for harder scrambles
            return 100 + efficiency_bonus + difficulty_bonus
        else:
            # Less harsh penalty, encourage exploration
            return -0.5
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([e[0] for e in batch])
        actions = torch.tensor([e[1] for e in batch]).to(self.device)
        rewards = torch.tensor([e[2] for e in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([e[3] for e in batch])
        dones = torch.tensor([e[4] for e in batch]).to(self.device)
        
        # Double DQN - use main network to select actions, target network to evaluate
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_actions = self.q_network(next_states).max(1)[1]
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze().detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Huber loss for more stable training
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory)
        }
        torch.save(checkpoint, f'checkpoint/checkpoint_epoch_{epoch}.pth')
        print(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # Check if checkpoint is compatible with current architecture
        try:
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
            print("Loaded compatible checkpoint")
        except RuntimeError as e:
            print(f"Checkpoint architecture mismatch: {e}")
            print("Starting fresh training with improved architecture...")
            # Don't load model weights, but keep other training progress
            
        # Load optimizer and training state
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("Optimizer state incompatible, using fresh optimizer")
            
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        
        # Load memory with new maxlen
        if 'memory' in checkpoint:
            old_memory = checkpoint['memory']
            self.memory = deque(old_memory, maxlen=20000)
        
        return checkpoint['epoch']

def train_agent(episodes=1000):
    agent = RubiksSolver()
    scores = deque(maxlen=100)
    
    # Check for existing checkpoints
    checkpoint_files = [f for f in os.listdir('checkpoint') if f.startswith('checkpoint_epoch_')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        start_epoch = agent.load_checkpoint(f'checkpoint/{latest_checkpoint}')
        print(f"Resumed training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("Starting training from scratch")
    
    # Initialize progress bar
    pbar = tqdm(range(start_epoch, episodes), desc="Training", 
                initial=start_epoch, total=episodes)
    
    # Statistics tracking
    last_checkpoint_time = time.time()
    episodes_per_sec = 0
    
    for episode in pbar:
        agent.current_episode = episode
        cube = Rubiks()
        
        # Curriculum learning - gradually increase scramble difficulty
        scramble_count = agent.get_curriculum_scrambles(episode)
        for _ in range(scramble_count):
            move = random.choice(cube.move_options)
            cube.move(move)
        
        state = agent.state_to_tensor(cube)
        total_reward = 0
        max_steps = 200
        
        for step in range(max_steps):
            action = agent.choose_action(state)
            move = cube.move_options[action]
            
            # Make move
            cube.move(move)
            next_state = agent.state_to_tensor(cube)
            
            # Improved reward calculation
            if cube.solved():
                reward = agent.calculate_improved_reward(cube, step, True, scramble_count)
                done = True
            else:
                reward = agent.calculate_improved_reward(cube, step, False, scramble_count)
                done = False
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        scores.append(total_reward)
        agent.replay()
        
        # Update target network every 100 episodes
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Save checkpoint every 10000 episodes
        if episode % 10000 == 0:
            agent.save_checkpoint(episode)
            current_time = time.time()
            if episode > start_epoch:
                episodes_per_sec = 10000 / (current_time - last_checkpoint_time)
                eta_hours = (episodes - episode) / episodes_per_sec / 3600
                pbar.set_postfix({
                    'Checkpoint': f'{episode}',
                    'EPS/sec': f'{episodes_per_sec:.1f}',
                    'ETA': f'{eta_hours:.1f}h'
                })
            last_checkpoint_time = current_time
        
        # Update progress bar every 100 episodes with current stats
        if episode % 100 == 0:
            avg_score = np.mean(scores) if scores else 0
            current_lr = agent.scheduler.get_last_lr()[0] if hasattr(agent.scheduler, 'get_last_lr') else agent.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Avg_Score': f'{avg_score:.1f}',
                'Epsilon': f'{agent.epsilon:.3f}',
                'Scrambles': f'{scramble_count}',
                'LR': f'{current_lr:.6f}'
            })
    
    pbar.close()
    print("Training completed!")

if __name__ == "__main__":
    print("Starting Rubik's Cube RL Training...")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    train_agent(episodes=2000000)
    print("Training completed!")