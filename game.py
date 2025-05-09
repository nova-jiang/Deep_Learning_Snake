import pygame
import sys
import time
from typing import List, Tuple, Optional
import numpy as np
from snake import Snake, Direction
from environment import Environment

class SnakeGame:
    COLORS = {
        'WHITE': (255, 255, 255),
        'SNAKE_HEAD': (0, 100, 0),
        'SNAKE_BODY': (50, 205, 50),
        'FOOD': (255, 0, 0),
        'OBSTACLE': (0, 0, 139),
        'WALL': (47, 79, 79),
        'GRID': (220, 220, 220),
        'TEXT': (0, 0, 0)
    }
    
    def __init__(self, size: Tuple[int, int] = (20, 20), cell_size: int = 30):
        """
        Game Initialization
        """
        pygame.init()

        self.size = size
        self.cell_size = cell_size
        self.width = size[1] * cell_size
        self.height = size[0] * cell_size
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game')
        
        self.env = Environment(size)
        self.snake = Snake()
        self.game_over = False
        self.game_speed = 10
        self.clock = pygame.time.Clock()

        self.episode = 0
        self.total_reward = 0
    
    def reset(self):
        self.env.reset()
        self.snake = Snake()
        self.game_over = False
        self.total_reward = 0
        self.episode += 1
    
    def draw(self):
        self.screen.fill(self.COLORS['WHITE']) # white background

        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, self.COLORS['GRID'], (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, self.COLORS['GRID'], (0, y), (self.width, y))

        for wall in self.env.walls:
            self._draw_cell(wall, self.COLORS['WALL'])

        for obstacle in self.env.obstacles:
            self._draw_cell(obstacle, self.COLORS['OBSTACLE'])
        
        self._draw_cell(self.env.food, self.COLORS['FOOD'])
        
        for i, segment in enumerate(self.snake.body):
            color = self.COLORS['SNAKE_HEAD'] if i == 0 else self.COLORS['SNAKE_BODY']
            self._draw_cell(tuple(segment), color)
        
        self._draw_training_info()

        pygame.display.flip()
    
    def _draw_cell(self, pos: Tuple[int, int], color: Tuple[int, int, int]):
        """
        Draw a single cell
        """
        rect = pygame.Rect(
            pos[1] * self.cell_size + 1,
            pos[0] * self.cell_size + 1,
            self.cell_size - 2,
            self.cell_size - 2
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
    
    def _draw_training_info(self):
        """
        Draw the training infomation
        """
        font = pygame.font.Font(None, 36)
        episode_text = font.render(f'Episode: {self.episode}', True, self.COLORS['TEXT'])
        self.screen.blit(episode_text, (10, 10))

        score_text = font.render(f'Score: {self.snake.score}', True, self.COLORS['TEXT'])
        self.screen.blit(score_text, (10, 50))

        reward_text = font.render(f'Total Reward: {self.total_reward}', True, self.COLORS['TEXT'])
        self.screen.blit(reward_text, (10, 90))
    
    def update(self, action: Direction) -> Tuple[float, bool]:
        """
        Args:
            action: the direction chose by AI
        Returns:
            Tuple[float, bool]: [rewards, end of game of not]
        """
        if self.game_over:
            return 0, True
        
        old_distance = self._get_distance_to_food()
        new_head = self.snake.move(action)
        new_distance = self._get_distance_to_food()
        
        reward = 0
        if self.snake.check_collision(self.env.walls) or \
           tuple(new_head) in self.env.obstacles:
            reward = -100
            self.game_over = True
            return reward, True
        
        if tuple(new_head) == self.env.food:
            self.snake.grow()
            self.env.food = self.env._generate_food()
            reward = 10
        else:
            reward = 0.1 if new_distance < old_distance else -0.1

        self.env.update_dynamic_obstacles() # call env function to update obstacles
        self.total_reward += reward
        return reward, False
    
    def _get_distance_to_food(self) -> float:
        """
        helper function
        """
        return abs(self.snake.head[0] - self.env.food[0]) + \
               abs(self.snake.head[1] - self.env.food[1])
    
    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    def run_training_step(self):
        self.check_events()
        self.draw()
        self.clock.tick(self.game_speed)

if __name__ == "__main__":
    game = SnakeGame()
    while True:
        game.check_events()
        game.draw()
        game.clock.tick(10)