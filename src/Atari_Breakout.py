import sys
import time
import pygame
import numpy as np

from enum import Enum
from math import cos, sin, pi
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.preprocessing import normalize as norm


class Colors(Enum):
    Black = (0, 0, 0, 255)
    Blue = (0, 100, 255, 255)
    Orange = (255, 180, 70, 255)
    Red = (255, 0, 0, 255)
    White = (255, 255, 255, 255)


class AtariGame:
    def __init__(self, _speed=1, _brickx=8, _bricky=5, _white=False, _banner=False, _screen=True):
        # Hyperparameters
        self.windowWidth = 400
        self.windowHeight = 300
        self.runspeed = _speed
        self.fps = 50
        self.fpsClock = pygame.time.Clock()
        self.brickx, self.bricky = _brickx, _bricky
        self.brick = np.ones((_bricky, _brickx), dtype=int)
        self.brickRect = [[pygame.Rect(self.windowWidth * j // _brickx, 20 * i + 40,
                                       (self.windowWidth * (j + 1) // _brickx) -
                                       (self.windowWidth * j // _brickx), 20)
                           for j in range(_brickx)] for i in range(_bricky)]
        self.ballPos = np.array([200.0, 200.0])
        self.ballSpeed = 5
        self.ballVector = norm([[-1, -1]]).ravel()
        self.ballSize = 8
        self.barLength = 50
        self.barWidth = 10
        self.barinitX, self.barinitY = 200.0, 295.0
        self.barRect = pygame.Rect(self.barinitX - self.barLength // 2, self.barinitY - self.barWidth // 2,
                                   self.barLength, self.barWidth)
        self.seconds = 0
        self.ticks = 0
        self.barSpeed = 8
        self.barMove = 0
        self.start = True
        self.end = False
        self.white = _white
        self.banner = _banner
        self.screen = _screen
        self.n_features = 5
        self.actions = 3

        self.ball_rand()
        pygame.init()
        self.surface = pygame.display.set_mode((self.windowWidth, self.windowHeight))
        pygame.display.set_caption('Atari Breakout')

    def get_state(self):
        a = np.empty(5, dtype=float)  # bricks[40] ballPos[2] ballVec[2] barX
        a[0] = self.ballPos[0]
        a[1] = self.ballPos[1]
        a[2] = self.ballVector[0]
        a[3] = self.ballVector[1]
        a[4] = self.barRect.centerx
        return a

    def die(self):
        self.draw()
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        if self.white:
            text = font.render('You lose!', True, Colors.Red)
        else:
            text = font.render('You lose!', True, Colors.White)
        text_rect = text.get_rect()  # Set the center of the text box
        text_rect.centerx = self.surface.get_rect().centerx
        text_rect.centery = self.surface.get_rect().centery
        self.surface.blit(text, text_rect)
        pygame.display.update()

    def gameclear(self):
        self.draw()
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        if self.white:
            text = font.render('You win!', True, Colors.Red)
        else:
            text = font.render('You Win!!', True, (255, 255, 255))
        text_rect = text.get_rect()  # Set the center of the text box
        text_rect.centerx = self.surface.get_rect().centerx
        text_rect.centery = self.surface.get_rect().centery
        self.surface.blit(text, text_rect)
        pygame.display.update()

    def timesup(self):
        self.draw()
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        if self.white:
            text = font.render('Time\'s up!', 1, Colors.Red)
        else:
            text = font.render('Time\'s up!', 1, Colors.White)
        text_rect = text.get_rect()  # Set the center of the text box
        text_rect.centerx = self.surface.get_rect().centerx
        text_rect.centery = self.surface.get_rect().centery
        self.surface.blit(text, text_rect)
        pygame.display.update()

    def restart(self):
        self.brick = np.ones((self.bricky, self.brickx), dtype=int)
        self.ballPos = np.array([200.0, 200.0])
        self.ball_rand()
        self.barRect = pygame.Rect(self.barinitX - self.barLength // 2, self.barinitY - self.barWidth // 2,
                                   self.barLength, self.barWidth)
        self.seconds = 0
        self.ticks = 0
        self.start = True
        self.end = False
        self.draw()
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        image_data = rgb2gray(resize(image_data, (80, 80), mode='reflect'))
        image_stack = np.stack([image_data for _ in range(2)])
        return image_stack

    def draw(self):
        self.surface.fill(Colors.Black.value)
        if self.white:
            pygame.draw.circle(self.surface, Colors.White.value, tuple(self.ballPos.astype(int)), self.ballSize)
        else:
            pygame.draw.circle(self.surface, Colors.Red.value, tuple(self.ballPos.astype(int)), self.ballSize)
        for i in range(self.bricky):
            for j in range(self.brickx):
                if self.brick[i][j] == 1:
                    if not self.white:
                        if (i + j) % 2 == 0:
                            rectcolor = Colors.Orange.value
                        else:
                            rectcolor = Colors.Blue.value
                    else:
                        rectcolor = Colors.White.value
                    pygame.draw.rect(self.surface, rectcolor, self.brickRect[i][j])
        pygame.draw.rect(self.surface, Colors.White.value, self.barRect)
        pygame.display.update()

    def startpause(self):
        fontsize = 50
        font = pygame.font.SysFont(None, fontsize)

        self.draw()
        if self.white:
            text = font.render('Ready?', 1, Colors.Red)
        else:
            text = font.render('Ready?', 1, (255, 255, 255, 255))
        text_rect = text.get_rect()  # Set the center of the text box
        text_rect.centerx = self.surface.get_rect().centerx
        text_rect.centery = self.surface.get_rect().centery
        self.surface.blit(text, text_rect)
        pygame.display.update()
        time.sleep(0.3)

    def ball_rand(self):
        angle = (np.random.rand() * 0.5 + 1.27) * pi
        self.ballVector = np.array([cos(angle), sin(angle)])

    def ball_move(self):
        tmp_reward = 0.01
        # move
        pre_pos = np.copy(self.ballPos)
        self.ballPos += self.ballSpeed * self.ballVector

        # touchwall
        if self.ballPos[0] <= self.ballSize:
            self.ballPos[0] = float(self.ballSize)
            self.ballVector[0] *= -1
        elif self.ballPos[0] >= (self.windowWidth - self.ballSize):
            self.ballPos[0] = self.windowWidth - self.ballSize
            self.ballVector[0] *= -1
        if self.ballPos[1] <= self.ballSize:
            self.ballPos[1] = float(self.ballSize)
            self.ballVector[1] *= -1
        elif self.ballPos[1] >= (self.windowHeight - self.ballSize):
            return True, -1

        # breakbrick
        for i in range(self.bricky):
            for j in range(self.brickx):
                if self.brick[i][j] == 1:
                    # touch from right
                    if (self.brickRect[i][j].right - self.ballSize) < self.ballPos[0] <= (
                                self.brickRect[i][j].right + self.ballSize) and pre_pos[0] > self.ballPos[0] and \
                                            self.brickRect[i][j].top <= self.ballPos[1] < self.brickRect[i][j].bottom:
                        self.ballVector[0] *= -1
                        self.brick[i][j] = 0
                    # touch from left
                    if (self.brickRect[i][j].right + self.ballSize) > self.ballPos[0] >= (
                                self.brickRect[i][j].left - self.ballSize) \
                            and pre_pos[0] < self.ballPos[0] and self.brickRect[i][j].bottom > self.ballPos[1] >= \
                            self.brickRect[i][j].top:
                        self.ballVector[0] *= -1
                        self.brick[i][j] = 0
                    # touch from bottom
                    if (self.brickRect[i][j].bottom - self.ballSize) < self.ballPos[1] <= (
                                self.brickRect[i][j].bottom + self.ballSize) \
                            and pre_pos[1] > self.ballPos[1] and self.brickRect[i][j].right > self.ballPos[0] >= \
                            self.brickRect[i][j].left:
                        self.ballVector[1] *= -1
                        self.brick[i][j] = 0
                    # touch from top
                    if (self.brickRect[i][j].top + self.ballSize) > self.ballPos[1] >= (
                                self.brickRect[i][j].top - self.ballSize) \
                            and pre_pos[1] < self.ballPos[1] and self.brickRect[i][j].right > self.ballPos[0] >= \
                            self.brickRect[i][j].left:
                        self.ballVector[1] *= -1
                        self.brick[i][j] = 0

        tmp_vector = np.copy(self.ballVector)

        # touchpad
        ball_rect = pygame.Rect(
            self.ballPos[0] - self.ballSize,
            self.ballPos[1] - self.ballSize,
            self.ballSize * 2,
            self.ballSize * 2
        )

        if pygame.Rect.colliderect(ball_rect, self.barRect):
            tmp_vector = norm([self.ballPos - np.array(self.barRect.center)]).ravel()
            if tmp_vector[1] >= 0:
                tmp_vector = norm([[tmp_vector[0], 1]]).ravel()
            elif tmp_vector[1] > -0.5:
                if tmp_vector[0] > 0:
                    tmp_vector += norm(np.array([[0.86, -0.5]])).ravel() - tmp_vector
                else:
                    tmp_vector += norm(np.array([[-0.86, -0.5]])).ravel() - tmp_vector
            tmp_reward = 0.1

        self.ballVector = tmp_vector

        if np.sum(self.brick) > 0:
            return False, tmp_reward
        else:
            return True, 1

    def move_bar(self):
        self.barRect.left += self.barSpeed * self.barMove
        if self.barRect.left >= 400 - self.barLength:
            self.barRect.left = int(400.0 - self.barLength)
        if self.barRect.left <= 0:
            self.barRect.left = 0

    def render(self, _move=0):
        # check valid
        if self.end:
            return
        terminal = False
        # startPause
        if self.start:
            if self.banner:
                self.startpause()
            self.start = False

        # draw things
        self.draw()

        # update
        if _move == 0:
            self.barMove = 0
        elif _move == 1:
            self.barMove = -1
        else:
            self.barMove = 1
        self.move_bar()
        preb = np.sum(self.brick)
        self.end, tmp_reward = self.ball_move()
        aftb = np.sum(self.brick)
        if not self.end and aftb < preb and aftb < self.brickx * self.bricky - 1 and preb < self.brickx * self.bricky:
            tmp_reward = 0.2 * (preb - aftb)

        # time step
        self.ticks += 1
        if self.ticks == self.fps:
            self.seconds, self.ticks = self.seconds + 1, 0
        if self.seconds == 180:
            self.end = True
            if self.banner:
                self.timesup()
            terminal = True

        # check gameover
        elif self.end:
            if self.banner:
                if np.sum(self.brick) > 0:
                    self.die()
                else:
                    self.gameclear()
            terminal = True

        pygame.event.clear()
        pygame.display.update()
        image_data = pygame.surfarray.array2d(pygame.display.get_surface())
        image_data = rgb2gray(resize(image_data, (80, 80), mode='reflect'))
        return image_data, tmp_reward, terminal

if __name__ == '__main__':
    # Game start
    atari = AtariGame(_speed=1, _brickx=16, _bricky=5, _white=False, _banner=True)
    end = False
    gamepoint = 0
    while True:
        # getkey
        pressedKeys = pygame.key.get_pressed()
        if pressedKeys[ord('q')]:
            pygame.quit()
            sys.exit()
        elif pressedKeys[ord('a')] and not pressedKeys[ord('d')]:
            move = 1
        elif pressedKeys[ord('d')] and not pressedKeys[ord('a')]:
            move = 2
        else:
            move = 0

        screen, reward, end = atari.render(_move=move)
        if reward > 1:
            gamepoint += reward
            print('Point:', gamepoint)
        elif reward < 0:
            gamepoint = 0
        if end:
            time.sleep(0.5)
            atari.restart()
        else:
            atari.fpsClock.tick(atari.fps * atari.runspeed)
