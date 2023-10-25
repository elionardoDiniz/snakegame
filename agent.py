import torch
import random
import numpy as np
from snake import SnakeGame, Point, Direction
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000

class Agent:
    def __init__(self):
        self.memory = deque(..., maxlen=MAX_MEMORY)
    
    def train_long_memory(self):
        pass
    
    def train_short_memory(self):
        pass

def train():
    pass

if __name__ == '__main__':
    train()