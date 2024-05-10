import numpy as np
from gymnasium import spaces
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, LEFT, DOWN, RIGHT, UP

from utils.utils import generate_random_map


class FrozenLake(FrozenLakeEnv):
    def __init__(self, **kwargs):
        self.truncate = kwargs.pop('truncate', False)
        self.render_fps = kwargs.pop('render_fps', 6)
        self.hole_penalty = kwargs.pop('hole_penalty', 0.0)
        self.random_reset = kwargs.pop('random_reset', False)
        self.frozen_prob = kwargs.pop('frozen_prob', 0.5)

        super().__init__(**kwargs)
        self.metadata['render_fps'] = self.render_fps

        self.observation_space = spaces.Dict({
            'current': spaces.MultiDiscrete([self.nrow, self.ncol]),
            'goal': spaces.MultiDiscrete([self.nrow, self.ncol]),
            'map': spaces.MultiBinary([self.nrow, self.ncol]),
        })

        # goal은 desc에서 G의 위치
        self.goal = []
        self.start = []
        self.goal = self._find_goal()
        self.start = self._find_start()
        self.current = self.start
        self.step_limit = self.ncol * self.nrow * 4
        self.n_step = 0

        self.map = [[True] * self.ncol for _ in range(self.nrow)]
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == b'H':
                    self.map[i][j] = False

    def _find_goal(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == b'G':
                    return i, j

    def _find_start(self):
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.desc[i][j] == b'S':
                    return i, j

    def step(self, a):
        obs, reward, done, truncated, info = super().step(a)
        self.current = (self.s // self.ncol, self.s % self.ncol)
        self.n_step += 1
        if self.truncate and self.n_step >= self.step_limit:
            done = True
            truncated = True
        if self.desc[self.current[0]][self.current[1]] == b'H':
            reward = self.hole_penalty
        return {'current': self.current, 'goal': self.goal, 'map': self.map}, reward, done, truncated, info

    def reset(self, **kwargs):
        if self.random_reset:
            self.desc = np.asarray(generate_random_map(self.ncol, self.nrow, p=self.frozen_prob), dtype='c')

            self.goal = self._find_goal()
            self.start = self._find_start()
            self.current = self.start

            nA = 4
            nS = self.nrow * self.ncol

            self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

            def to_s(_row, _col):
                return _row * self.ncol + _col

            def inc(_row, _col, a):
                if a == LEFT:
                    _col = max(_col - 1, 0)
                elif a == DOWN:
                    _row = min(_row + 1, self.nrow - 1)
                elif a == RIGHT:
                    _col = min(_col + 1, self.ncol - 1)
                elif a == UP:
                    _row = max(_row - 1, 0)
                return _row, _col

            def update_probability_matrix(_row, _col, action):
                newrow, newcol = inc(_row, _col, action)
                newstate = to_s(newrow, newcol)
                newletter = self.desc[newrow, newcol]
                terminated = bytes(newletter) in b"GH"
                reward = float(newletter == b"G")
                return newstate, reward, terminated

            self.map = [True] * self.nrow * self.ncol

            for row in range(self.nrow):
                for col in range(self.ncol):
                    s = to_s(row, col)
                    for a in range(4):
                        li = self.P[s][a]
                        letter = self.desc[row, col]
                        if letter in b"GH":
                            li.append((1.0, s, 0, True))
                            if letter == b'H':
                                self.map[row * self.ncol + col] = False
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

            reset = super().reset(**kwargs)
            self.s = self.current[0] * self.ncol + self.current[1]
            list_reset = list(reset)
            list_reset[0] = int(self.s)
            return tuple(list_reset)

        else:
            self.current = self.start
            self.n_step = 0
            return {'current': self.current, 'goal': self.goal, 'map': self.map}