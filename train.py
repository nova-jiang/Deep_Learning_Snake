import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from game import SnakeGame
from agent import DQNAgent
from snake import Direction
import json

class SnakeTrainer:
    def __init__(
        self,
        state_size: int = 14,
        episodes_per_stage: list[int] = [500, 300, 200], #training stages config
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.997,
        checkpoint_dir: str = "checkpoints",
        stats_dir: str = "stats"
    ):
        self.game = SnakeGame()
        self.agent = DQNAgent(
            state_size=state_size,
            batch_size=128,
            learning_rate=0.0005,
            gamma=0.95
        )
        
        self.episodes_per_stage = episodes_per_stage
        self.total_episodes = sum(episodes_per_stage)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.checkpoint_dir = checkpoint_dir
        self.stats_dir = stats_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_losses': [],
            'scores': [],
            'moving_avg_reward': []
        }

    def save_stats(self):
        stats_file = f"{self.stats_dir}/training_stats.json"
        
        json_stats = {
            'episode_rewards': [float(x) for x in self.stats['episode_rewards']],
            'episode_lengths': [int(x) for x in self.stats['episode_lengths']],
            'scores': [int(x) for x in self.stats['scores']],
            'moving_avg_reward': [float(x) for x in self.stats['moving_avg_reward']] if self.stats['moving_avg_reward'] else [],
            'episode_losses': [float(x) for x in self.stats['episode_losses']] if self.stats['episode_losses'] else [],
            'training_params': {
                'total_episodes': self.total_episodes,
                'episodes_per_stage': self.episodes_per_stage,
                'final_epsilon': float(self.epsilon),
                'max_score': max(self.stats['scores']),
                'avg_score': float(np.mean(self.stats['scores'])),
                'avg_reward': float(np.mean(self.stats['episode_rewards']))
            }
        }
        with open(stats_file, 'w') as f:
            json.dump(json_stats, f, indent=4)
        
        self.plot_training_progress(f"{self.stats_dir}/training_progress.png")
    
    def plot_training_progress(self, save_path=None):
        """
        For all the training plots
        """
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        episodes = range(len(self.stats['episode_rewards']))
        plt.plot(episodes, self.stats['episode_rewards'], label='Episode Reward', alpha=0.6)
        if self.stats['moving_avg_reward']:
            plt.plot(episodes, self.stats['moving_avg_reward'], 
                    label='Moving Average (100 episodes)', 
                    color='red', linewidth=2)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(episodes, self.stats['scores'], color='green')
        plt.title('Score per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(episodes, self.stats['episode_lengths'], color='orange')
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)
    
        if self.stats['episode_losses']:
            plt.subplot(2, 2, 4)
            loss_episodes = range(len(self.stats['episode_losses']))
            plt.plot(loss_episodes, self.stats['episode_losses'], color='red')
            plt.title('Average Loss per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def update_stats(self, episode_reward, steps, episode_loss):

        self.stats['episode_rewards'].append(episode_reward)
        self.stats['episode_lengths'].append(steps)
        self.stats['scores'].append(self.game.snake.score)
        if episode_loss:
            self.stats['episode_losses'].append(np.mean(episode_loss))
        
        window_size = min(100, len(self.stats['episode_rewards']))
        moving_avg = np.mean(self.stats['episode_rewards'][-window_size:])
        self.stats['moving_avg_reward'].append(moving_avg)
        
        total_episodes = len(self.stats['episode_rewards'])
        if total_episodes % 100 == 0:
            self.save_stats()
            self.plot_training_progress()
    
    def print_progress(self, total_episode, episode_reward, steps, start_time):
        episode_time = time.time() - start_time
        print(f"Episode {total_episode + 1}/{self.total_episodes} - "
              f"Score: {self.game.snake.score} - "
              f"Reward: {episode_reward:.2f} - "
              f"Steps: {steps} - "
              f"Epsilon: {self.epsilon:.3f} - "
              f"Time: {episode_time:.2f}s")
    
    def save_final_results(self):
        self.save_stats()
        #save our model
        final_model_path = f"{self.checkpoint_dir}/final_model.pth"
        self.agent.save(final_model_path)
        
        print("\nTraining Final Result:")
        print(f"total_episodes: {self.total_episodes}")
        print(f"best_score: {max(self.stats['scores'])}")
        print(f"average_score: {np.mean(self.stats['scores']):.2f}")
        print(f"epsilon: {self.epsilon:.3f}")
        
        stages = ['Env Difficulty 0', 'Env Difficulty 1', 'Env Difficulty 2']
        start_idx = 0
        for stage, episodes in zip(stages, self.episodes_per_stage):
            end_idx = start_idx + episodes
            stage_scores = self.stats['scores'][start_idx:end_idx]
            print(f"\n{stage}stage:")
            print(f"episodes: {episodes}")
            print(f"best_scores: {max(stage_scores)}")
            print(f"average_score: {np.mean(stage_scores):.2f}")
            start_idx = end_idx
    
    def train_stage(self, episodes: int, stage: int, difficulty: float):
        """
        Training for each stage
        """
        print(f"\nStage {stage + 1} Training - Difficulty: {difficulty:.2f}")
        self.game.env.set_difficulty(difficulty)
        
        for episode in range(episodes):
            start_time = time.time()
            
            # reset env
            self.game.reset()
            state = self.agent.get_state(self.game)
            episode_reward = 0
            episode_loss = []
            steps = 0
            
            while not self.game.game_over:
                # choose an action
                action = self.agent.select_action(state, self.epsilon)
                
                # act
                reward, done = self.game.update(action)
                next_state = self.agent.get_state(self.game)
                
                reward = self.agent.get_state_reward(self.game, action, done)
                
                self.agent.store_experience(state, action, reward, next_state, done)
                
                if loss := self.agent.train():
                    episode_loss.append(loss)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                self.game.run_training_step()
            
            # update epsilon
            self.epsilon = max(self.epsilon_end, 
                             self.epsilon * self.epsilon_decay)
            
            self.update_stats(episode_reward, steps, episode_loss)

            total_episode = sum(self.episodes_per_stage[:stage]) + episode
            self.print_progress(total_episode, episode_reward, steps, start_time)
    
    def train(self):
        print("Start Training...")
        
        # Stage 0: static env
        self.train_stage(self.episodes_per_stage[0], 0, 0.0)
        
        # Stage 1: 0.3 difficulty level dynamic env
        self.train_stage(self.episodes_per_stage[1], 1, 0.3)
        
        # Stage 2：complete dynamic env
        self.train_stage(self.episodes_per_stage[2], 2, 1.0)
        
        print("\n Training Complete！")
        self.save_final_results()

    def evaluate(self, model_path: str, n_episodes: int = 20):
        """
        Evaluation for each stage
        """
        print(f"\nEvaluation Start: {model_path}")

        self.agent.load(model_path)
        
        eval_stats = {
            'scores': [],
            'steps': [],
            'rewards': []
        }
        
        # end epsilon
        original_epsilon = self.epsilon
        self.epsilon = 0
        
        for episode in range(n_episodes):
            self.game.reset()
            state = self.agent.get_state(self.game)
            total_reward = 0
            steps = 0
            
            while not self.game.game_over:
                action = self.agent.select_action(state, epsilon=0)

                reward, done = self.game.update(action)
                next_state = self.agent.get_state(self.game)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                self.game.run_training_step()
            
            eval_stats['scores'].append(self.game.snake.score)
            eval_stats['steps'].append(steps)
            eval_stats['rewards'].append(total_reward)



if __name__ == "__main__":
    trainer = SnakeTrainer(
        episodes_per_stage=[500, 300, 200], #500, 300, 200
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.997
    )
    trainer.train()
    trainer.evaluate("checkpoints/best_model.pth", n_episodes=20)