import numpy as np
from collections import deque
from enum import Enum

class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Snake:
    def __init__(self, start_pos=(5, 5), initial_length=3):
        """
        Initialize the snake
        Args:
            start_pos: starting position of (5,5)
            initial_length: initial length of 3
        """
        self.head = np.array(start_pos)
        self.direction = Direction.RIGHT
        
        # using deque to store the snake
        self.body = deque([np.array((start_pos[0], start_pos[1] - i)) 
                          for i in range(initial_length)])
        
        self.is_alive = True
        self.score = 0
        self.steps_without_food = 0
        self.food_value = 50

    def move(self, new_direction=None):
        if new_direction is not None:
            # preventing the move to opposite directions
            if abs(self.direction.value - new_direction.value) != 2:
                self.direction = new_direction
    
        movement = {
            Direction.UP: np.array((-1, 0)),
            Direction.RIGHT: np.array((0, 1)),
            Direction.DOWN: np.array((1, 0)),
            Direction.LEFT: np.array((0, -1))
        }
        
        new_head = self.head + movement[self.direction]
        self.body.appendleft(new_head)
        self.head = new_head
        self.body.pop()
        self.steps_without_food += 1
        
        return new_head
    
    def grow(self):
        """
        Snake grow by one when get food
        """
        self.body.append(self.body[-1]) # append to the tail
        self.score += self.food_value
        self.steps_without_food = 0
    
    def check_collision(self, walls):
        # check if collide with walls
        if tuple(self.head) in walls:
            self.is_alive = False
            return True
            
        # check if collide with snake body
        body_positions = [tuple(pos) for pos in list(self.body)[1:]]
        if tuple(self.head) in body_positions:
            self.is_alive = False
            return True
            
        return False
    
    def get_state(self):
        """
        get current snake state
        """
        return {
            'head_position': self.head,
            'body_positions': list(self.body),
            'direction': self.direction,
            'is_alive': self.is_alive,
            'score': self.score,
            'steps_without_food': self.steps_without_food
        }