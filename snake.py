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
SPEED = 14

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.score = 0
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.RIGHT
        self.head = [self.w / 2, self.h / 2]
        self.snake = [self.head, 
                        Point(self.head[0] - BLOCK_SIZE, self.head[1]),
                        Point(self.head[0] - (2 * BLOCK_SIZE), self.head[1])]
        self._place_food()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        if __name__ == '__main__':
            self._handle_input()
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return game_over, self.score
    
    def _is_collision(self):
        if self.head[0] > self.w - BLOCK_SIZE or self.head[0] < 0 or self.head[1] > self.h - BLOCK_SIZE or self.head[1] < 0 or self.head in self.snake[1:]:
            return True
        return False
    
    def _move(self, direction):
        x = self.head[0]
        y = self.head[1]
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
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt[0]+4, pt[1]+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
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
        game_over, score = game.play_step()
        if game_over == True:
            break
    print(f'Final Score: {score}')
    pygame.quit()