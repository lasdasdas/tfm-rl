
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
import pygame
import time

from PIL import Image
import PIL

scale = 35
render_op = 1


class Board():
    def __init__(self, height=10, width=10):
        self.h = height
        self.w = width
        self.apples = []
        self.apple_prob = 0.1
        # init

    def randPoint(self):
        y = random.randint(0, self.h - 1)
        x = random.randint(0, self.w - 1)
        return (x, y)

    def reset(self):
        self.apples = []


def sumT(t1, t2):
    return(t1[0]+t2[0], t1[1]+t2[1])


def absT(t):
    return(abs(t[0]), abs(t[1]))


class Snake:
    def __init__(self, board):
        self.board = board
        # tracking queue
        self.deq = deque()
        # tracking matrix
        self.mat = np.zeros((board.w, board.h))
        # direction
        self.dir = (1, 0)
        self.finish = False
        self.score = 0

        # init
        init_point = (board.w // 2, board.h // 2)
        self.deq.append(init_point)
        self.insertToMat(init_point)
        self.steps = 0

    def insertToMat(self, pt):
        self.mat[self.board.h - 1 - pt[1], pt[0]] = 1

    def removeFromMat(self, pt):
        self.mat[self.board.h - 1 - pt[1], pt[0]] = 0

    def updateDir(self, new_dir):
        assert abs(sum(new_dir)) == 1
        if sum(absT(sumT(self.dir, new_dir))) == 0:
            return
        self.dir = new_dir

    def updateTail(self):
        self.removeFromMat(self.deq[0])
        self.deq.popleft()

    def updateHead(self, nh):
        self.insertToMat(nh)
        self.deq.append(nh)

    def updateApples(self, nh):
        if nh in self.board.apples:
            self.board.apples = [i for i in self.board.apples if i != nh]
            return True
        if (len(self.board.apples) == 0):
            # or random.random() < self.board.apple_prob):
            count = 0
            while True:
                count = count + 1
                rpt = self.board.randPoint()
                if rpt not in self.deq:
                    self.board.apples.append(rpt)
                    break
                if count > 50:
                    break
        return False

    def collision(self, nh):
        if (nh[0] >= self.board.w or
                nh[1] >= self.board.h or
                nh[0] < 0 or nh[1] < 0):
            return True

        if nh in self.deq:
            return True
        return False

    def step(self, new_dir=None):
        if new_dir is None:
            new_dir = self.dir
        # Check of new head
        head = self.deq[-1]
        self.updateDir(new_dir)
        new_head = sumT(head, self.dir)
        self.finish = self.collision(new_head)

        self.steps += 1
        if self.finish:
            self.score = -1
        else:
            self.updateHead(new_head)
            has_apples = self.updateApples(new_head)
            if has_apples:
                self.score = 2
                # This is tmp
                # self.updateTail()
            else:
                self.score = -0.01
                self.updateTail()
        return self.getState(), self.score, self.finish, self.steps

    def draw(self, display):
        ns = pygame.Surface((self.board.w, self.board.h))
        ns.fill((55, 155, 255))
        # Draw snake
        for e in self.deq:
            rect = pygame.Rect(e[0], e[1], 1, 1)
            pygame.draw.rect(ns, (0, 0, 155),  rect)

        for e in self.board.apples:
            rect = pygame.Rect(e[0], e[1], 1, 1)
            pygame.draw.rect(ns, (0, 255, 0),  rect)

        ns = pygame.transform.scale(ns,
                                    [scale*self.board.h, scale*self.board.w])
        ns = pygame.transform.flip(ns, False, True)
        display.blit(ns, (0, 0))
        pygame.display.flip()

    def draw_frame(self, display, frame):
        ns = pygame.surfarray.make_surface(frame)
        ns = pygame.transform.scale(ns,
                                    [scale*self.board.h, scale*self.board.w])
        ns = pygame.transform.flip(ns, False, True)
        display.blit(ns, (0, 0))
        pygame.display.flip()

    def getState(self):
        ret = np.zeros((self.board.w, self.board.h, 4),  dtype=np.uint8)
        # Body
        ret[:, :, 0] = self.mat*255
        # Head
        head = self.deq[-1]
        ret[self.board.h - 1 - head[1], head[0], 1] = 255
        ret[self.board.h - 1 - head[1], head[0], 0] = 0
        if self.dir == (0, -1):
            ret[0, :, 3] = 255
        elif self.dir == (0, 1):
            ret[-1, :, 3] = 255
        elif self.dir == (1, 0):
            ret[:, 0, 3] = 255
        elif self.dir == (-1, 0):
            ret[:, -1, 3] = 255

        # Apples
        for a in self.board.apples:
            ret[self.board.h - 1 - a[1], a[0], 2] = 255
        return ret


class Environ():
    def __init__(self, board=Board(5,5)):
        self.board_size= board.w
        self.action_size = 4
        self.num_actions = 4
        self.s = Snake(board)
        self.b = board
        pygame.init()
        if render_op:
            self.scale = 40
            size = (board.w*self.scale, board.h*self.scale)
            self.display = pygame.display.set_mode(size)

    def reset(self):
        self.b.reset()
        self.s = Snake(self.b)
        return self.s.getState()

    def step(self, action):
        if action == 0:
            return self.s.step((0, 1))
        if action == 1:
            return self.s.step((0, -1))
        if action == 2:
            return self.s.step((1, 0))
        if action == 3:
            return self.s.step((-1, 0))

    def render(self):
        frame_1 = self.s.getState()
        frame = np.zeros((self.b.w, self.b.h, 3),  dtype=np.uint8)
        for i, pt in enumerate(self.s.deq):
            if i+1 == len(self.s.deq):
                frame[self.s.board.h - 1 - pt[1], pt[0], 1] = 200
            else:
                frame[self.s.board.h - 1 - pt[1], pt[0], 0] = 200

        frame[:, :, 2] = frame_1[:, :, 2]  # + 0.5* frame_1[:,:,3]
        # plt.imshow(frame)
        # plt.draw()
        # plt.pause(0.001)
        ns = pygame.surfarray.make_surface(frame)
        ns = pygame.transform.scale(ns,
                                    [self.scale*frame.shape[0], self.scale*frame.shape[1]])
        for i in range(len(self.s.deq) - 1):
            pt1 = self.s.deq[i]
            pt2 = self.s.deq[i+1]

            pt1corr = (self.scale * (self.s.board.h - 1 -
                       pt1[1]) + self.scale/2, pt1[0] * self.scale + self.scale/2)
            pt2corr = (self.scale * (self.s.board.h - 1 -
                       pt2[1]) + self.scale/2, pt2[0] * self.scale + self.scale/2)
            print(pt1corr)
            pygame.draw.line(ns, (255, 255, 255), pt1corr, pt2corr, width=6)
        # ns = pygame.transform.flip(ns, False, True)
        ns = pygame.transform.rotate(ns, 90)
        self.display.blit(ns, (0, 0))
        pygame.display.flip()

    def render_frame(self, frame):
        if render_op:
            self.s.draw_frame(self.display, frame)

    def sample(self):
        return random.randrange(4)
