import pygame
import random

# 화면 크기 설정
WIDTH = 800
HEIGHT = 600

# 색깔 정의
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# 패들 설정
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
PADDLE_SPEED = 10

# 공 설정
BALL_RADIUS = 10
BALL_SPEED_X = 5
BALL_SPEED_Y = 5

# 벽돌 설정
BRICK_WIDTH = 100
BRICK_HEIGHT = 30
BRICK_ROWS = 5
BRICK_COLS = 8
BRICK_SPACING = 10

# 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Breakout")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

def game_loop():
    start_time = pygame.time.get_ticks()  # 게임 시작 시간
    score = 0  # 스코어

    # 패들 생성
    paddle = pygame.Rect((WIDTH - PADDLE_WIDTH) // 2, HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)

    # 공 생성
    ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS, HEIGHT // 2 - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)

    # 벽돌 생성
    bricks = []
    for i in range(BRICK_ROWS):
        for j in range(BRICK_COLS):
            brick = pygame.Rect(j * (BRICK_WIDTH + BRICK_SPACING), i * (BRICK_HEIGHT + BRICK_SPACING) + 50, BRICK_WIDTH, BRICK_HEIGHT)
            bricks.append(brick)

    ball_speed_x = BALL_SPEED_X * random.choice([1, -1])
    ball_speed_y = BALL_SPEED_Y * random.choice([1, -1])

    restart_game = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, 0
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    restart_game = True
                elif event.key == pygame.K_ESCAPE:
                    return False, 0

        if restart_game:
            break

        # 패들 이동
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and paddle.left > 0:
            paddle.x -= PADDLE_SPEED
        if keys[pygame.K_RIGHT] and paddle.right < WIDTH:
            paddle.x += PADDLE_SPEED

        # 게임중 종료
        if keys[pygame.K_ESCAPE]:
            pygame.quit()

        # 공 이동
        ball.x += ball_speed_x
        ball.y += ball_speed_y

        # 벽과 공 충돌 처리
        if ball.left <= 0 or ball.right >= WIDTH:
            ball_speed_x *= -1
        if ball.top <= 0:
            ball_speed_y *= -1

        # 패들과 공 충돌 처리
        if ball.colliderect(paddle):
            ball_speed_y *= -1
            score += 1

        # 벽돌과 공 충돌 처리
        for brick in bricks[:]:
            if ball.colliderect(brick):
                bricks.remove(brick)
                ball_speed_y *= -1
                score += 10

        # 게임 오버 처리
        if ball.bottom >= HEIGHT:
            return True, score

        # 게임 화면 그리기
        screen.fill(BLACK)
        pygame.draw.rect(screen, BLUE, paddle)
        pygame.draw.circle(screen, YELLOW, ball.center, BALL_RADIUS)
        for brick in bricks:
            pygame.draw.rect(screen, WHITE, brick)

        # 게임 시간 표시
        elapsed_time = pygame.time.get_ticks() - start_time
        time_text = font.render("Time: " + str(round(elapsed_time / 1000, 1)) + "s", True, WHITE)
        screen.blit(time_text, (10, 10))

        # 스코어 표시
        score_text = font.render("Score: " + str(score), True, WHITE)
        screen.blit(score_text, (WIDTH - 150, 10))

        pygame.display.flip()

        # 게임 속도 조절
        clock.tick(60)

    return restart_game, score

# 게임 루프 실행
while True:
    game_over, score = game_loop()
    if not game_over:
        continue
    elif game_over == True:
        game_over_text = font.render("Game Over! Score: " + str(score) + ". Press 'R' to restart or 'ESC' to quit", True, WHITE)
        screen.blit(game_over_text, ((WIDTH - game_over_text.get_width()) // 2, (HEIGHT - game_over_text.get_height()) // 2))
        pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    waiting = False
                    pygame.quit()
                    quit()

