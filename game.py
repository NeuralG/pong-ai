import pygame
import random

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class PongGame:
    def __init__(self, w=640, h=640):

        self.w = w
        self.h = h

        pygame.init()
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption("AI Pong")
        self.clock = pygame.time.Clock()

        self.paddle = pygame.Rect(self.w // 2, self.h - 25, 100, 15)
        self.ball = pygame.Rect(self.w // 2, self.h // 2, 20, 20)

        self.ballDx = 4
        self.ballDy = 6

        self.score = 0
        self.reward = 0
        self.isGameOver = False
        self.isColliding = False

        self.font = pygame.font.SysFont("arial", 25)

    def moveRight(self):
        self.paddle.x += 5
        if self.paddle.x > self.w - 100:
            self.paddle.x = self.w - 100

    def moveLeft(self):
        self.paddle.x -= 5
        if self.paddle.x < 0:
            self.paddle.x = 0

    def moveBall(self):
        self.ball.x += self.ballDx
        self.ball.y += self.ballDy
        if self.ball.left < 0: 
            self.ballDx *= -1
            self.ball.left = 0 
        elif self.ball.right > self.w: 
            self.ballDx *= -1
            self.ball.right = self.w 
        if self.ball.y < 0:
            self.ballDy *= -1
            self.ball.top = 0
        if self.ball.y >= self.h:
            self.gameOver()
        if self.isColliding:
            if self.ball.right < self.paddle.left or self.ball.left > self.paddle.right:
                self.isColliding = False

    def handleCollision(self):

        def calcReward():
            base = 15 # base reward
            comboBonus = self.score - 1 # combo reward
            return base + comboBonus

        if self.isColliding:
            return

        intersection = self.ball.clip(self.paddle)

        if intersection.width > intersection.height:
            self.ballDy *= -1
            self.ball.bottom = self.paddle.top
            self.score += 1
            self.reward = calcReward()
        else:
            self.ballDx *= -1

    def gameOver(self):
        print(f"Score: {self.score}")
        self.isGameOver = True
        self.reward = -20

    def playStep(self, action):

        def calcReward():
            paddleCenter = self.paddle.x + 100 // 2
            ballCenter = self.ball.x + 20 // 2
            dist = abs(paddleCenter-ballCenter)
            maxDist = self.w 
            
            return 0.8 - (dist / maxDist)


        self.reward = calcReward()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.moveBall()
        if action == [0, 1, 0]:
            self.moveRight()
        elif action == [0, 0, 1]:
            self.moveLeft()
        if self.ball.colliderect(self.paddle):
            self.handleCollision()

        self.display.fill(BLACK)
        text = self.font.render(f"Score: {self.score}", True, GREEN)
        self.display.blit(text, [0, 0])
        pygame.draw.rect(self.display, WHITE, self.paddle)
        pygame.draw.rect(self.display, RED, self.ball)
        pygame.display.flip()
        self.clock.tick(60)

        return (self.reward, self.score, self.isGameOver)

    def reset(self):
        self.ballDx = 4
        self.ballDy = 6
        self.score = 0
        self.paddle.x, self.paddle.y = self.w // 2, self.h - 25
        self.ball.x, self.ball.y = self.w // 2, self.h // 2
        self.isGameOver = False

    def getState(self):
        paddleCenter = self.paddle.x + 100 // 2 
        ballCenter = self.ball.x + 20 // 2
        gameCenterX = self.w // 2

        arr = [
            paddleCenter / self.w, #normalized paddle x
            ballCenter / self.w, #normalized ball x
            (ballCenter - paddleCenter) / self.w, #distance x
            self.ballDx / 4.5, # speed x
            self.ballDy / 6, # speed y
            self.ball.y / self.h, # ball y
            (paddleCenter - gameCenterX) / self.w
        ]
        return arr

if __name__ == "__main__":
    game = PongGame()
    while True:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            action = [0, 1, 0] 
        elif keys[pygame.K_LEFT]:
            action = [0, 0, 1] 
        else:
            action = [1, 0, 0]
        game.playStep(action)
