# FrozenLake_custom_env.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import closing
from io import StringIO
from typing import List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.utils import seeding
from gymnasium.envs.registration import register

import random  

# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Predefined Maps
MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

def is_valid(board: List[List[str]], max_size: int) -> bool:
    """DFS to check there's a path from S(0,0) to G ignoring holes."""
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if (r, c) not in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                nr, nc = r + x, c + y
                if not (0 <= nr < max_size and 0 <= nc < max_size):
                    continue
                if board[nr][nc] == "G":
                    return True
                if board[nr][nc] != "H":  # 'F' or 'S'
                    frontier.append((nr, nc))
    return False

def generate_random_map_one_bear(size=5, p_F=0.9, seed=None) -> List[str]:
    """
    Generates a random valid map with exactly one 'B'.
    'F' = floor, 'H' = hole, 'B' = single bear tile, 'S' = start, 'G' = goal
    """
    np_random, _ = seeding.np_random(seed)
    valid = False
    board = []

    while not valid:
        base = np_random.choice(["F", "H"], (size, size), p=[p_F, 1 - p_F])
        base[0][0] = "S"
        base[-1][-1] = "G"
        if is_valid(base, size):
            valid = True
            board = base

    # place exactly 1 bear
    possible_positions = []
    for r in range(size):
        for c in range(size):
            if (r, c) != (0, 0) and (r, c) != (size - 1, size - 1):
                if board[r][c] == "F":
                    possible_positions.append((r, c))

    if possible_positions:
        br, bc = np_random.choice(possible_positions)
        board[br][bc] = "B"

    return ["".join(row) for row in board]

class CustomFrozenLake(gym.Env):
    """
    Base FrozenLake environment that ensures exactly one 'B'.
    """
    metadata = {"render_modes": ["human", "ansi", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, desc=None, map_name=None, is_slippery=True):
        if desc is None and map_name is None:
            # Generate a single-bear map
            desc = generate_random_map_one_bear(size=5, p_F=0.9)
        elif desc is None:
            desc = MAPS[map_name]

        self.desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = self.desc.shape

        nA = 4
        nS = self.nrow * self.ncol
        self.reward_range = (0, 1)
        self.render_mode = render_mode

        # initial distribution
        starts = (self.desc == b"S").astype(float).ravel()
        self.initial_state_distrib = starts / starts.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        self.is_slippery = is_slippery

        def to_s(r, c):
            return r * self.ncol + c

        def inc(r, c, a):
            if a == LEFT:
                c = max(c - 1, 0)
            elif a == DOWN:
                r = min(r + 1, self.nrow - 1)
            elif a == RIGHT:
                c = min(c + 1, self.ncol - 1)
            elif a == UP:
                r = max(r - 1, 0)
            return (r, c)

        def update_prob(r, c, a):
            # Hole => -3 => done
            # Goal => +10 => done
            nr, nc = inc(r, c, a)
            ns = to_s(nr, nc)
            letter = self.desc[nr][nc]
            if letter == b"H":
                return ns, -3, True
            elif letter == b"G":
                return ns, 10, True
            return ns, 0, False

        # Build transition table
        nA = 4
        for r in range(self.nrow):
            for c in range(self.ncol):
                s = to_s(r, c)
                letter = self.desc[r][c]
                for a in range(nA):
                    li = self.P[s][a]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b_ in [(a - 1) % 4, a, (a + 1) % 4]:
                                prob = 1.0 / 3.0
                                ns, rew, done = update_prob(r, c, b_)
                                li.append((prob, ns, rew, done))
                        else:
                            ns, rew, done = update_prob(r, c, a)
                            li.append((1.0, ns, rew, done))

        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)

        # internal state
        self.s = None
        self.lastaction = None

        # Pygame
        self._setup_pygame()

    def _setup_pygame(self):
        self.window_size = (min(64 * self.ncol, 512), min(64 * self.nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None
        self.polar_bear_img = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        np_random, _ = seeding.np_random(seed)
        self.s = categorical_sample(self.initial_state_distrib, np_random)
        self.lastaction = None
        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0}

    def step(self, action):
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], np.random)
        p, s_, r, done = transitions[i]
        self.s = s_
        self.lastaction = action
        if self.render_mode == "human":
            self.render()
        return int(s_), r, done, False, {"prob": p}

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        import pygame
        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Single-Bear Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        from os import path
        if self.ice_img is None:
            f_name = path.join(path.dirname(__file__), "/home/xgang/Desktop/img/ice.png")
            self.ice_img = pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
        if self.hole_img is None:
            f_name = path.join(path.dirname(__file__), "/home/xgang/Desktop/img/hole.png")
            self.hole_img = pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
        if self.cracked_hole_img is None:
            f_name = path.join(path.dirname(__file__), "/home/xgang/Desktop/img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
        if self.goal_img is None:
            f_name = path.join(path.dirname(__file__), "/home/xgang/Desktop/img/goal.png")
            self.goal_img = pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
        if self.start_img is None:
            f_name = path.join(path.dirname(__file__), "/home/xgang/Desktop/img/stool.png")
            self.start_img = pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
        if self.polar_bear_img is None:
            f_name = "/home/xgang/Desktop/img/icons8-polar-bear-64.png"
            self.polar_bear_img = pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "/home/xgang/Desktop/img/elf_left.png"),
                path.join(path.dirname(__file__), "/home/xgang/Desktop/img/elf_down.png"),
                path.join(path.dirname(__file__), "/home/xgang/Desktop/img/elf_right.png"),
                path.join(path.dirname(__file__), "/home/xgang/Desktop/img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(e), self.cell_size) for e in elfs
            ]

        desc = self.desc.tolist()
        # fill ice first
        for r in range(self.nrow):
            for c in range(self.ncol):
                pos = (c*self.cell_size[0], r*self.cell_size[1])
                self.window_surface.blit(self.ice_img, pos)
        # then holes, bear, goal, start
        import pygame
        for r in range(self.nrow):
            for c in range(self.ncol):
                pos = (c*self.cell_size[0], r*self.cell_size[1])
                tile = desc[r][c]
                if tile == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif tile == b"B":
                    self.window_surface.blit(self.polar_bear_img, pos)
                elif tile == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif tile == b"S":
                    self.window_surface.blit(self.start_img, pos)
                rect = (*pos, *self.cell_size)
                pygame.draw.rect(self.window_surface, (180,200,230), rect, 1)

        # agent
        row = self.s // self.ncol
        col = self.s % self.ncol
        agent_pos = (col*self.cell_size[0], row*self.cell_size[1])
        last_a = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_a]
        if desc[row][col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, agent_pos)
        else:
            self.window_surface.blit(elf_img, agent_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _render_text(self):
        from io import StringIO
        out = StringIO()
        r, c = self.s//self.ncol, self.s%self.ncol
        desc = self.desc.tolist()
        desc = [[ch.decode("utf-8") for ch in line] for line in desc]
        desc[r][c] = utils.colorize(desc[r][c], "red", highlight=True)
        if self.lastaction is not None:
            out.write(f"  ({['Left','Down','Right','Up'][self.lastaction]})\n")
        else:
            out.write("\n")
        out.write("\n".join("".join(row) for row in desc)+"\n")
        return out.getvalue()

    def close(self):
        import pygame
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()


class CustomFrozenLakePolarBear(CustomFrozenLake):

    def __init__(self, render_mode=None, desc=None, map_name=None,
                 is_slippery=True, max_steps=50):
        super().__init__(render_mode=render_mode, desc=desc, map_name=map_name, is_slippery=is_slippery)
        self.max_steps = max_steps
        self.current_step = 0

        # find the single bear (r, c)
        self.bear = None
        self.goal = None
        for r in range(self.nrow):
            for c in range(self.ncol):
                if self.desc[r][c] == b"B":
                    self.bear = (r, c)
                if self.desc[r][c] == b"G":
                    self.goal = (r, c)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.current_step = 0
        self.bear = None
        for r in range(self.nrow):
            for c in range(self.ncol):
                if self.desc[r][c] == b"B":
                    self.bear = (r, c)
        return obs, info

    def step(self, action):
        # agent moves first
        self.current_step += 1
        s_next, base_r, base_done, _, info = super().step(action)
        row_a = s_next // self.ncol
        col_a = s_next % self.ncol

        # if hole or goal => done
        if base_done:
            # base_r = base_r - self.current_step
            return s_next, base_r, True, False, info

        # if agent meets bear => -5 => done
        if self.bear == (row_a, col_a):
            return s_next, -5.0, True, False, info

        # bear moves randomly every step
        if self.bear is not None:
            (br, bc) = self.bear
            (br2, bc2) = self._move_bear_randomly(br, bc)
            # remove old 'B', put 'F'
            if self.desc[br][bc] == b"B":
                self.desc[br][bc] = b"F"
            self.desc[br2][bc2] = b"B"
            self.bear = (br2, bc2)

            # check collision again
            if (row_a, col_a) == self.bear:
                return s_next, -5.0, True, False, info

        # truncated if max steps reached
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
        return s_next, base_r, False, truncated, info

    def _move_bear_randomly(self, br, bc):
        import random
        directions = [(1,0),(0,1),(-1,0),(0,-1)]
        random.shuffle(directions)
        for d_r, d_c in directions:
            nr = br + d_r
            nc = bc + d_c
            # Skip out of bounds or S/G
            if 0 <= nr < self.nrow and 0 <= nc < self.ncol:
                if self.desc[nr][nc] not in [b"S", b"G", b"H"]:
                    return (nr, nc)
        return (br, bc)

    def _agent_near_goal(self, r, c):
        gr, gc = self.goal
        # Updated condition to check if agent is within 4x4 of goal
        return abs(r - gr) <= 4 and abs(c - gc) <= 4


class PartialObservationWrapper(gym.ObservationWrapper):
    """
    Partial local window. If bear is in window => 2.
    """
    def __init__(self, env: gym.Env, window_size=1):
        super().__init__(env)
        self.env = env
        self.window_size = window_size
        obs_h = 2*window_size + 1
        obs_w = 2*window_size + 1
        self.observation_space = spaces.Box(low=0, high=4, shape=(obs_h, obs_w), dtype=np.int32)

    def observation(self, obs: int):
        row = obs // self.env.ncol
        col = obs % self.env.ncol
        w = self.window_size
        out_h = 2*w+1
        out_w = 2*w+1
        grid = np.zeros((out_h, out_w), dtype=np.int32)

        for i in range(-w, w+1):
            for j in range(-w, w+1):
                rr = row + i
                cc = col + j
                if rr<0 or rr>=self.env.nrow or cc<0 or cc>=self.env.ncol:
                    val = 0  # out of bounds => floor
                else:
                    tile = self.env.desc[rr][cc]
                    if tile == b"H":
                        val = 1
                    elif tile == b"B":
                        val = 2
                    elif tile == b"G":
                        val = 3
                    elif tile == b"S":
                        val = 4
                    else:
                        val = 0  # 'F'
                grid[i+w, j+w] = val
        return grid

# Register base env + polar-bear env
register(
    id='CustomFrozenLake-v0',
    entry_point='FrozenLake_custom_env:CustomFrozenLake',
    max_episode_steps=200,
    reward_threshold=1.0,
)
register(
    id='CustomFrozenLakePolarBear-v0',
    entry_point='FrozenLake_custom_env:CustomFrozenLakePolarBear',
    max_episode_steps=500,
)
