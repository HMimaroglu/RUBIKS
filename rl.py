import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
from rubiks import Rubiks

# Create checkpoint directory
os.makedirs('checkpoint', exist_ok=True)

class DQN(nn.Module):
    def __init__(self, input_size=54, hidden_size=512, output_size=8):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class RubiksSolver:
    def __init__(self, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN().to(self.device)
        self.target_network = DQN().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.memory = deque(maxlen=10000)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 32
        self.gamma = 0.95
        
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
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.stack([e[0] for e in batch])
        actions = torch.tensor([e[1] for e in batch]).to(self.device)
        rewards = torch.tensor([e[2] for e in batch]).to(self.device)
        next_states = torch.stack([e[3] for e in batch])
        dones = torch.tensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
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
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = deque(checkpoint['memory'], maxlen=10000)
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
    
    for episode in range(start_epoch, episodes):
        cube = Rubiks()
        # Scramble with limited moves for training
        for _ in range(random.randint(1, 10)):
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
            
            # Calculate reward
            if cube.solved():
                reward = 100  # Large reward for solving
                done = True
            else:
                reward = -1   # Small penalty for each step
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
        
        # Print progress
        if episode % 10 == 0:
            avg_score = np.mean(scores) if scores else 0
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    print("Starting Rubik's Cube RL Training...")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    train_agent(episodes=2000000)
    print("Training completed!")