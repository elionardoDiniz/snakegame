import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('Arial.ttf')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 2000

class SnakeGame:
    
    def __init__(self, w=800, h=800):
        self.w = w
        self.h = h
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        
        # init game state
        self.reset()
    
    def reset(self):
        self.score = 0
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, 
                        Point(self.head.x - BLOCK_SIZE, self.head.y),
                        Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self._place_food()
        self.frame_iterations = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        reward = 0
        self.frame_iterations += 1
        if __name__ == '__main__':
            self._handle_input()
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        game_over = False
        if self._is_collision() or self.frame_iterations > 60 * len(self.snake):
            game_over = True
            reward -= 10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            self._place_food()
            reward += 10
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
    
    def _is_collision(self, pt = None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0 or pt in self.snake[1:]:
            return True
        return False
    
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        moves = {
            Direction.RIGHT: lambda: (x + BLOCK_SIZE, y),
            Direction.LEFT: lambda: (x - BLOCK_SIZE, y),
            Direction.UP: lambda: (x, y - BLOCK_SIZE),
            Direction.DOWN: lambda: (x, y + BLOCK_SIZE),
        }
        (x, y) = moves[direction]()
        self.head = Point(x, y)
    
    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _handle_input(self):
        directions = {
            pygame.K_LEFT: Direction.LEFT,
            pygame.K_RIGHT: Direction.RIGHT,
            pygame.K_UP: Direction.UP,
            pygame.K_DOWN: Direction.DOWN,
        }
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN and event.key in directions:
                self.direction = directions[event.key]
    
    def _apply_action(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
    
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clockwise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            self.direction = clockwise[idx]
        if np.array_equal(action, [0, 1, 0]):
            self.direction = clockwise[(idx + 1) % 4]
        if np.array_equal(action, [0, 0, 1]):
            self.direction = clockwise[(idx - 1) % 4]

if __name__ == '__main__':
    game = SnakeGame()

    while True:
        game.play_step()
        reward, game_over, score = game.play_step()
        if game_over == True:
            break
    print(f'Final Score: {score}')
    pygame.quit()