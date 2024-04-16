import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# 환경 설정
WIDTH, HEIGHT = 800, 600
WHITE, BLACK, BLUE, YELLOW = (255, 255, 255), (0, 0, 0), (0, 0, 255), (255, 255, 0)
PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED = 100, 20, 10
BALL_RADIUS, BALL_SPEED_X, BALL_SPEED_Y = 10, 5, 5
BRICK_WIDTH, BRICK_HEIGHT, BRICK_ROWS, BRICK_COLS, BRICK_SPACING = 100, 30, 5, 8, 10

# 게임 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Breakout")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Deep Q-Network 모델 정의
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN 에이전트 정의
class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.input_size = input_size
        self.output_size = output_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.output_size)
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32))).item()
            else:
                target = reward
            target_f = self.model(torch.tensor(state, dtype=torch.float32)).clone().detach()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(torch.tensor(state, dtype=torch.float32)), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 게임 환경 클래스 정의
class Breakout:
    def __init__(self):
        self.paddle = pygame.Rect((WIDTH - PADDLE_WIDTH) // 2, HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
        self.bricks = [pygame.Rect(j * (BRICK_WIDTH + BRICK_SPACING), i * (BRICK_HEIGHT + BRICK_SPACING) + 50, BRICK_WIDTH, BRICK_HEIGHT)
                       for i in range(BRICK_ROWS) for j in range(BRICK_COLS)]
        self.ball_speed_x = BALL_SPEED_X * random.choice([1, -1])
        self.ball_speed_y = BALL_SPEED_Y * random.choice([1, -1])
        self.agent = DQNAgent(input_size=4, output_size=2)

    def get_state(self):
        return [self.ball.left, self.ball.top, self.ball_speed_x, self.ball_speed_y]

    def reset(self):
        self.paddle = pygame.Rect((WIDTH - PADDLE_WIDTH) // 2, HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
        self.bricks = [pygame.Rect(j * (BRICK_WIDTH + BRICK_SPACING), i * (BRICK_HEIGHT + BRICK_SPACING) + 50, BRICK_WIDTH, BRICK_HEIGHT)
                       for i in range(BRICK_ROWS) for j in range(BRICK_COLS)]
        self.ball_speed_x = BALL_SPEED_X * random.choice([1, -1])
        self.ball_speed_y = BALL_SPEED_Y * random.choice([1, -1])
        return self.get_state()

    def step(self, action):
        reward = 0
        if action == 0:  # 왼쪽으로 이동
            self.paddle.x -= PADDLE_SPEED
        elif action == 1:  # 오른쪽으로 이동
            self.paddle.x += PADDLE_SPEED

        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        if self.ball.left <= 0 or self.ball.right >= WIDTH:
            self.ball_speed_x *= -1
        if self.ball.top <= 0:
            self.ball_speed_y *= -1

        if self.ball.colliderect(self.paddle):
            self.ball_speed_y *= -1
            reward += 1

        for brick in self.bricks[:]:
            if self.ball.colliderect(brick):
                self.bricks.remove(brick)
                self.ball_speed_y *= -1
                reward += 10

        if self.ball.bottom >= HEIGHT:
            done = True
        else:
            done = False

        return self.get_state(), reward, done

# 게임 환경 초기화
game_env = Breakout()

# 총 보상을 저장할 리스트
total_rewards = []

# 학습
for episode in range(1000):  # 적절한 에피소드 수를 선택해야 함
    state = game_env.reset()
    done = False
    total_reward = 0
    while not done:
        # # 게임 플레이 화면 업데이트
        # for event in pygame.event.get():
        #     print(event)
        #     print(event.type)
        #     if event.type == pygame.QUIT:
        #         done = True
        #         break
        # if done:
        #     break
        
        action = game_env.agent.choose_action(state)
        next_state, reward, done = game_env.step(action)
        game_env.agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        game_env.agent.replay(batch_size=32)  # DQN을 사용한 학습

        # 게임 화면 그리기
        screen.fill(BLACK)
        pygame.draw.rect(screen, BLUE, game_env.paddle)
        pygame.draw.circle(screen, YELLOW, game_env.ball.center, BALL_RADIUS)
        for brick in game_env.bricks:
            pygame.draw.rect(screen, WHITE, brick)
        pygame.display.flip()
        clock.tick(600)

    total_rewards.append(total_reward)
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # 학습 과정 시각화
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.pause(0.05)  # 업데이트를 시각화하기 위해 잠시 멈춤