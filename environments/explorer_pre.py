import numpy as np
import pygame

scale = 35


class Environ():
    def __init__(self, render=True):
        self.board_size = 7

        self.rewards = 12
        self.action_size = 4

        self.num_actions = 4

        self.bad_score = -4
        self.neutral_score = -0.5
        self.discover_score = -0.5
        self.good_score = 1

        self.render_op = render

        if self.render_op:
            self.scale = 40
            size = (self.board_size*self.scale, self.board_size*self.scale)
            self.display = pygame.display.set_mode(size)

        self.borders = np.ones((self.board_size + 4, self.board_size + 4))
        self.borders[2:-2, 2:-2] = 0

        self.gmap = None

        self.reset()

    def create_map(self):
        self.gmap = None
        if self.gmap is None:
            r = np.zeros((self.board_size, self.board_size))

            gp = np.random.choice(self.board_size**2,
                                  self.rewards, replace=False)
            for i in range(self.rewards):
                r[gp[i] % self.board_size, gp[i] // self.board_size] = 1

            self.gmap = r

        return self.gmap.copy()

    def reset(self, ):
        self.map = self.create_map()
        self.explored = np.zeros((self.board_size, self.board_size))
        self.pos = np.random.randint(self.board_size, size=(2,))
        self.map[self.pos[0], self.pos[1]] = 0
        return self.get_state()

    def get_state(self):
        q = np.zeros((self.board_size, self.board_size, 3))
        q[:, :, 0] = self.map
        q[:, :, 1] = self.explored
        q[self.pos[0], self.pos[1], 2] = 1
        return q

    def render(self):
        if not self.render_op:
            return
        q = self.get_state()
        frame = q[:, :, 0] * 80 + q[:, :, 1] * 20 + q[:, :, 2] * 40
        ns = pygame.surfarray.make_surface(frame)
        ns = pygame.transform.scale(ns,
                                    [self.scale*frame.shape[0],
                                     self.scale*frame.shape[1]])
        ns = pygame.transform.rotate(ns, 90)
        ns = pygame.transform.flip(ns, False, True)
        self.display.blit(ns, (0, 0))
        pygame.display.flip()

    def step(self, action):
        s = self.get_state()

        p = None
        if action == 0:
            p = self.pos + np.array([1, 0])
        if action == 1:
            p = self.pos + np.array([0, 1])
        if action == 2:
            p = self.pos + np.array([-1, 0])
        if action == 3:
            p = self.pos + np.array([0, -1])

        # Outer colision
        if np.any((p >= self.board_size)) or np.any(p < 0):
            return s, self.bad_score, False, None

        # Update pose if not outter colision
        self.pos = p

        # If good
        if self.map[self.pos[0], self.pos[1]] == 1:
            self.map[self.pos[0], self.pos[1]] = 0
            self.explored[self.pos[0], self.pos[1]] = 1
            return s, self.good_score, False, None

        # No good
        ns = 0
        if self.explored[self.pos[0], self.pos[1]] == 0:
            ns = self.discover_score
        else:
            ns = self.neutral_score
        self.explored[self.pos[0], self.pos[1]] = 1

        finished = not np.any(self.map == 1)
        return self.get_state(), ns, finished, None

    def sample(self):
        return np.random.randint(self.num_actions)


if __name__ == "__main__":
    e = Environ()
    for i in range(500000000):
        print(i)
        act = e.sample()
        next_state, reward, done, info = e.step(act)
        if done:
            e.reset()
        else:
            e.render()

