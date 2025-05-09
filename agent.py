import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from model import DQN
from snake import Direction

# The Snake Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNAgent:
    """
    Deep Q Learning Agent
    """
    def __init__(
        self,
        state_size: int,
        memory_size: int = 100000,
        batch_size: int = 128,
        gamma: float = 0.95,
        learning_rate: float = 0.0005,
        target_update: int = 10
    ):
        self.state_size = state_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        
        self.policy_net = DQN(state_size)
        self.target_net = DQN(state_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.memory = deque(maxlen=memory_size)
        
        self.steps_done = 0
        self.last_score = 0
        self.last_food_distance = float('inf')
        self.food_reward = 50

        self.episode_rewards = []
        self.current_episode_reward = 0
    
    def get_food_distance(self, game) -> float:
        """
        Get the distance between snake head and food
        """
        head_x, head_y = game.snake.head
        food_x, food_y = game.env.food
        return abs(head_x - food_x) + abs(head_y - food_y)
    
    def get_state_reward(self, game, action, done) -> float:
        """
        Get rewards for the state
        """
        reward = 0
        
        # Besic reward
        reward += 0.2
        
        # Rewards for food. 50 per food.
        if game.snake.score > self.last_score:
            reward += self.food_reward
            self.last_score = game.snake.score
        
        # Rewards for getting closer to food
        current_distance = self.get_food_distance(game)
        if current_distance < self.last_food_distance:
            reward += 1
        elif current_distance > self.last_food_distance:
            reward -= 0.5
        self.last_food_distance = current_distance
        
        # Rewards for surviving in the game and stepping
        reward += 0.01 * game.snake.steps_without_food
        
        # Punishment for death
        if done:
            reward -= 10
        
        return reward
    
    def get_state(self, game):
        """
        Get current state from environment
        """
        # get head position
        head_x, head_y = game.snake.head
        
        # get food position
        food_x, food_y = game.env.food
        
        # get danger conditions from eight directions
        danger = np.zeros(8)
        directions = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
        
        for i, (dx, dy) in enumerate(directions):
            pos = (head_x + dx, head_y + dy)
            danger[i] = 1 if (pos in game.env.walls or 
                            pos in game.env.obstacles or 
                            pos in [tuple(p) for p in list(game.snake.body)[1:]]) else 0
        
        # Create the state
        state = np.concatenate([
            danger,
            [food_x - head_x, food_y - head_y],
            [int(game.snake.direction == Direction.UP),
             int(game.snake.direction == Direction.RIGHT),
             int(game.snake.direction == Direction.DOWN),
             int(game.snake.direction == Direction.LEFT)]
        ])
        
        return state
    
    def select_action(self, state, epsilon: float):
        """
        Select actions using ε-greedy
        """
        if random.random() < epsilon:
            return Direction(random.randint(0, 3))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return Direction(torch.argmax(q_values).item())
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action.value, reward, next_state, done))
        self.current_episode_reward += reward
    
    def end_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch.unsqueeze(1) + \
                            (1 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
        
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1)
        self.optimizer.step()
        
        if self.steps_done % self.target_update == 0:
            with torch.no_grad():
                for target_param, policy_param in zip(
                    self.target_net.parameters(), 
                    self.policy_net.parameters()
                ):
                    target_param.data.copy_(
                        0.95 * target_param.data + 0.05 * policy_param.data
                    )
        
        self.steps_done += 1
        return loss.item()
    
    def save(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']