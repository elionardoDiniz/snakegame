import torch
import random
import numpy as np
import json
from os.path import exists
from snake import SnakeGame, Point, Direction
from model import Linear_QNet, QTrainer
from collections import deque
from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.gamma = 0.9
        self.epsilon = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        if exists(r'./model/model.pth'):
            self.model.load_state_dict(torch.load(r'./model/model.pth'))
            self.model.eval()
            print('Model loaded')

    def get_state(self, game: SnakeGame):
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT

        head = game.snake[0]
        pt_up = Point(head.x, head.y - 20)
        pt_down = Point(head.x, head.y + 20)
        pt_left = Point(head.x - 20, head.y)
        pt_right = Point(head.x + 20, head.y)
        
        state = [
            (dir_up and game._is_collision(pt_up)) or
            (dir_down and game._is_collision(pt_down)) or
            (dir_left and game._is_collision(pt_left)) or
            (dir_right and game._is_collision(pt_right)),
            
            (dir_up and game._is_collision(pt_right)) or
            (dir_down and game._is_collision(pt_left)) or
            (dir_left and game._is_collision(pt_up)) or
            (dir_right and game._is_collision(pt_down)),
            
            (dir_up and game._is_collision(pt_left)) or
            (dir_down and game._is_collision(pt_right)) or
            (dir_left and game._is_collision(pt_down)) or
            (dir_right and game._is_collision(pt_up)),
            
            
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]
        
        return np.array(state, dtype=int)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        move = 0
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
        final_move[move] = 1
        return final_move
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    if exists(r'./model/infos.json'):
        with open(r'./model/infos.json') as info:
            infos = json.load(info)
            record = infos['Record']
            agent.n_games = infos['Games']
        
    while True:
        old_state = agent.get_state(game)
        final_move = agent.get_action(old_state)
        game._apply_action(final_move)
        reward, done, score = game.play_step()
        new_state = agent.get_state(game)
        agent.train_short_memory(old_state, final_move, reward, new_state, done)
        agent.remember(old_state, final_move, reward, new_state, done)
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
            if score > record:
                record = score
                agent.model.save()
            
            print(f'Game: {agent.n_games}, Score: {score}, Record: {record}')
            infos = {
                'Record': record,
                'Games': agent.n_games
            }
            with open('./model/infos.json', 'w') as info:
                json.dump(infos, info, indent = 4)

if __name__ == '__main__':
    train()