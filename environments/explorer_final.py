from python_tsp.exact import solve_tsp_dynamic_programming
import numpy as np
import time
import copy
import pygame

scale = 35

class Environ():
    def __init__(self, render=True):
        self.render_op = render
        self.board_size = 12

        self.board_size_ext = 12

        self.action_size = 4

        self.num_actions = 4

        self.display = None

        self.borders = np.ones((self.board_size + 4, self.board_size + 4))
        self.borders[2:-2, 2:-2] = 0

        self.reset()

    def create_map(self):
        self.bad_score = -1
        self.neutral_score = -0.1
        self.discover_score = 0.1
        self.good_score = 1

        self.good_pix = 8
        r = np.zeros((self.board_size, self.board_size))
        gp = np.random.choice(self.board_size**2, self.good_pix, replace=False)

        for i in range(self.good_pix):
            r[gp[i] % self.board_size, gp[i] // self.board_size] = 1

        return r

    def best_pol(self):
        target = self.solution[0]
        if (self.pos == target).all():
            self.solution = self.solution[1:]
            target = self.solution[0]

        assert(len(self.solution))
        if target[0] != self.pos[0]:
            if target[0] > self.pos[0]:
                return 0
            else:
                return 2
        elif target[1] > self.pos[1]:
            return 1
        else:
            return 3

    def greedy_pol(self):
        target = self.solution[0]
        if (self.pos == target).all():
            self.solution = self.solution[1:]
            target = self.solution[0]

        assert(len(self.solution))
        if target[0] != self.pos[0]:
            if target[0] > self.pos[0]:
                return 0
            else:
                return 2
        elif target[1] > self.pos[1]:
            return 1
        else:
            return 3

    @staticmethod
    def greedy_pol_unknown(st):

        s = np.where(st[:, :, 0] > 0.99)

        if len(s[0]) == 0:
            s = np.where(st[:, :, 1] < 0.99)

        curr_pose = np.where(st[:, :, 2] > 0.99)

        d = np.abs(s[0] - curr_pose[0][0]) + np.abs(s[1] - curr_pose[1][0])

        target_index = np.argmin(d)
        target = (s[0][target_index], s[1][target_index])

        pos = (curr_pose[0][0], curr_pose[1][0])

        if target[0] != pos[0]:
            if target[0] > pos[0]:
                return 0
            else:
                return 2
        elif target[1] > pos[1]:
            return 1
        else:
            return 3

    def discover(self):
        find = 2
        xmin = max(self.pos[0] - find, 0)
        xmax = min(self.pos[0] + find, self.board_size)

        ymin = max(self.pos[1] - find, 0)
        ymax = min(self.pos[1] + find, self.board_size)
        self.discovered[xmin:xmax, ymin:ymax] = 1

    def reset(self, ):
        self.diplay = None

        self.map = self.create_map()
        self.discovered = np.zeros(
            (self.board_size, self.board_size), dtype=bool)

        self.explored = np.zeros((self.board_size, self.board_size))
        while 1:
            self.pos = np.random.randint(self.board_size, size=(2,))
            if self.map[self.pos[0], self.pos[1]] == 0:
                break

        self.is_redundant = 0
        self.steps = 0
        self.discover()
        return self.get_state()

    def get_state(self):
        q = np.zeros((self.board_size, self.board_size, 3))
        q[:, :, 0] = np.multiply(self.map, self.discovered)
        q[:, :, 1] = self.discovered
        q[self.pos[0], self.pos[1], 2] = 1
        return q

    def render(self):
        if not self.render_op:
            return

        if self.render_op and self.display is None:
            self.scale = 40
            size = (self.board_size*self.scale, self.board_size*self.scale)
            self.display = pygame.display.set_mode(size)

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
        if self.is_redundant:
            return self.get_state(), 0, 1, True
        self.steps += 1

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
            return s, self.bad_score, False, False

        self.discover()
        # Update pose if not outter colision
        self.pos = p

        finished = not np.any(self.map > 0.5)

        # If good
        if self.map[self.pos[0], self.pos[1]] == 1:
            self.map[self.pos[0], self.pos[1]] = 0
            self.explored[self.pos[0], self.pos[1]] = 1
            finished = not np.any(self.map > 0.5)
            if finished:
                self.is_redundant = 1
            return s, self.good_score, finished, False

        # No good
        ns = 0
        if self.explored[self.pos[0], self.pos[1]] == 0:
            ns = self.discover_score
        else:
            ns = self.neutral_score
        self.explored[self.pos[0], self.pos[1]] = 1

        # self.as_graph()
        return self.get_state(), ns, finished, False

    def sample(self):
        return np.random.randint(self.num_actions)

    def solve_greedy_unknown(self):
        neoenv = self.copy()
        count = 0
        score = 0
        state = neoenv.get_state()
        while 1:
            act = Environ.greedy_pol_unknown(state)
            next_state, reward, done, info = neoenv.step(act)
            count += 1
            score += reward
            if done:
                break
            else:
                state = next_state

        return None, score, count

    def copy(self):
        # cannot deep copy pygame display(referece will live in the self object)
        self.display = None
        return copy.deepcopy(self)

    def solve_greedy(self):
        s = np.where(self.map > 0.99)

        xcoor = s[0].tolist()
        ycoor = s[1].tolist()
        distance = 0

        pt = [self.pos[0], self.pos[1]]
        ld = [np.abs(xcoor[i] - pt[0]) + np.abs(ycoor[i]-pt[1])
              for i in range(len(xcoor))]
        goal_pts = []

        for j in range(self.good_pix):
            imin = np.argmin(ld)
            pt[0] = xcoor[imin]
            pt[1] = ycoor[imin]
            distance += ld[imin]
            goal_pts.append(np.array(pt))

            xcoor.pop(imin)
            ycoor.pop(imin)
            ld = [np.abs(xcoor[i] - pt[0]) + np.abs(ycoor[i]-pt[1])
                  for i in range(len(xcoor))]

        assert(len(goal_pts) == self.good_pix)
        approx_best = self.good_pix * self.good_score\
            - distance*self.discover_score

        return goal_pts, approx_best, distance

    def solve(self):
        s = np.where(self.map == 1)

        xcoor = [self.pos[0]] + s[0].tolist()
        ycoor = [self.pos[1]] + s[1].tolist()

        num_nodes = len(xcoor)

        M = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                c = np.abs(xcoor[i] - xcoor[j]) + np.abs(ycoor[i] - ycoor[j])
                M[i, j] = c

        coords = []
        for i, j in zip(xcoor, ycoor):
            coords.append(np.array([i, j]))
        distance_matrix = M + M.transpose()
        distance_matrix[:, 0] = 0
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
        assert(distance > 0)

        approx_best = (len(xcoor) - 1) * self.good_score - \
            distance*self.discover_score

        coords = []
        for i in permutation:
            coords.append(np.array([xcoor[i], ycoor[i]]))
        return coords[1:], approx_best, distance


if __name__ == "__main__":
    e = Environ()
    state = e.reset()
    for i in range(500000000):
        time.sleep(0.01)
        act = Environ.greedy_pol_unknown(state)

        next_state, reward, done, info = e.step(act)
        state = next_state
        e.render()
        if done:
            print(e.steps)
            print("---------------------")
            e.reset()
            _, _, best_greedy = e.solve_greedy()
            _, _, best_greedy_uk = e.solve_greedy_unknown()
            _, _, best_tsp = e.solve()
            print("greedy ", best_greedy)
            print("greedy uk ", best_greedy_uk)
            print("tsp ", best_tsp)
