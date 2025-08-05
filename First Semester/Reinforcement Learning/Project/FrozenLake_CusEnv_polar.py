from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding
from gymnasium.envs.registration import register

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

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


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
                if board[r_new][c_new] in ["F", "B"]:
                    frontier.append((r_new, c_new))

    return False


def generate_random_map(size: int = 8, p_F: float = 0.85, p_B: float = 0.05, seed: Optional[int] = None) -> List[str]:

    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p_F = min(1, p_F)
        p_B = min(1, p_B)
        board = np_random.choice(["F", "H", "B"], (size, size), p=[p_F, 1 - p_F - p_B, p_B])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


class CustomFrozenLake(Env):

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, desc=None, map_name=None, is_slippery=True,):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]

        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)


        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            new_row, new_col = inc(row, col, action)
            new_state = to_s(new_row, new_col)
            new_letter = desc[new_row, new_col]
            terminated = bytes(new_letter) in b"BGH"
            reward = float(new_letter == b"G")
            return new_state, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
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
        self.wildcat_img = None

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return int(s), r, t, False, {"prob": p}

    def _move_polar_bears(self):
        new_positions = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

        for row, col in self.polar_bear_positions:
            # Randomly choose a valid direction
            np.random.shuffle(directions)
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.nrow and 0 <= new_col < self.ncol:
                    tile = self.desc[new_row][new_col].decode("utf-8")
                    if tile not in ["H", "G", "S"] and (new_row, new_col) not in new_positions:
                        new_positions.append((new_row, new_col))
                        break
            else:
                # If no valid move is found, stay in the same position
                new_positions.append((row, col))

        self.polar_bear_positions = new_positions


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[toy-text]"`'
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.polar_bear_img is None:
            file_name = '/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/icons8-polar-bear-64.png'
            self.polar_bear_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.wildcat_img is None:
            file_name = '/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/wildcat.jpg'
            self.wildcat_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/elf_left.png"),
                path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/elf_down.png"),
                path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/elf_right.png"),
                path.join(path.dirname(__file__), "/home/xgang/XGang/Graduation/First_Year/Reinforcement-Learning/Project/img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"B":
                    self.window_surface.blit(self.polar_bear_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)

        # paint the elf
        bot_row, bot_col = self.s // self.ncol, self.s % self.ncol
        cell_rect = (bot_col * self.cell_size[0], bot_row * self.cell_size[1])
        last_action = self.lastaction if self.lastaction is not None else 1
        elf_img = self.elf_images[last_action]

        if desc[bot_row][bot_col] == b"H":
            self.window_surface.blit(self.cracked_hole_img, cell_rect)
        elif desc[bot_row][bot_col] == b"B":
            self.window_surface.blit(self.wildcat_img, cell_rect)
        else:
            self.window_surface.blit(elf_img, cell_rect)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )


    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

register(
    id='CustomFrozenLake-v0',  # Unique environment ID
    entry_point='FrozenLake_CusEnv_polar:CustomFrozenLake',  # Python module path and class
    max_episode_steps=200,  # Optional: maximum steps per episode
    reward_threshold=1.0,  # Optional: threshold for considering the task solved
)

gym.pprint_registry()
env = gym.make('CustomFrozenLake-v0', render_mode='human')


