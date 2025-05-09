import numpy as np
from typing import List, Tuple, Set
import random

class Environment:
    def __init__(self, size: Tuple[int, int] = (20, 20), n_obstacles: int = 5, difficulty: float = 0):
        """
        Setting up game environment
        Args:
            size: the size of map
            n_obstacles: num of obstacles
            difficulty: the dynamic difficulty of the map. 
                        It varies from 0 to 1. 0 means static env and 1 means very difficult dynamic env
        """
        self.size = size
        self.n_obstacles = n_obstacles
        self.difficulty = max(0, min(1, difficulty))
        self.reset()
    
    def set_difficulty(self, difficulty: float):
        self.difficulty = max(0, min(1, difficulty))
    
    def update_dynamic_obstacles(self):
        """
        update obstacles according to dynamic level
        """
        update_probability = 0.1 * self.difficulty
        
        if random.random() < update_probability:
            # remove
            if self.obstacles and random.random() < 0.5:
                self.obstacles.remove(random.choice(list(self.obstacles)))
            
            # add
            available_positions = [
                (i, j) for i in range(1, self.size[0]-1) 
                for j in range(1, self.size[1]-1)
                if (i, j) not in self.walls and 
                   (i, j) not in self.obstacles and 
                   (i, j) != self.food
            ]
            
            if available_positions:
                self.obstacles.add(random.choice(available_positions))
    
    def reset(self):
        self.grid = np.zeros(self.size)
        self.walls = set(self._generate_walls())
        self.obstacles = set(self._generate_obstacles())
        self.food = self._generate_food()
        self.steps = 0
        
    def _generate_walls(self) -> Set[Tuple[int, int]]:
        walls = set()
        for i in range(self.size[0]):
            walls.add((i, 0))
            walls.add((i, self.size[1]-1))
        for j in range(self.size[1]):
            walls.add((0, j))
            walls.add((self.size[0]-1, j))
        return walls
    
    def _generate_obstacles(self) -> Set[Tuple[int, int]]:
        obstacles = set()
        available_positions = [
            (i, j) for i in range(1, self.size[0]-1) 
            for j in range(1, self.size[1]-1)
        ]
        
        # remove obstacles in the center space
        center = (self.size[0]//2, self.size[1]//2)
        available_positions = [
            pos for pos in available_positions 
            if abs(pos[0]-center[0]) > 3 or abs(pos[1]-center[1]) > 3
        ]
        
        for _ in range(self.n_obstacles):
            if available_positions:
                pos = random.choice(available_positions)
                obstacles.add(pos)
                available_positions.remove(pos)
        
        return obstacles
    
    def _generate_food(self) -> Tuple[int, int]:
        """
        Randomly generate food
        """
        available_positions = [
            (i, j) for i in range(1, self.size[0]-1) 
            for j in range(1, self.size[1]-1)
            if (i, j) not in self.walls and (i, j) not in self.obstacles
        ]
        return random.choice(available_positions)
    
    def get_state(self):
        return {
            'walls': self.walls,
            'obstacles': self.obstacles,
            'food': self.food,
            'size': self.size,
            'steps': self.steps
        }
    
    def is_collision(self, position: Tuple[int, int]) -> bool:
        return position in self.walls or position in self.obstacles
    
    def is_food(self, position: Tuple[int, int]) -> bool:
        return position == self.food