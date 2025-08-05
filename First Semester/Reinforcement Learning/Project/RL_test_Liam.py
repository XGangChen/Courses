# RL_test.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import pygame

class FrozenLakeWithBearEnv(gym.Env):
    """
    A custom environment that:
    1. Randomly generates holes on a grid each episode.
    2. If the agent (human) steps on a hole, there's a 25% chance a polar bear is spawned
       from one of the holes to chase the agent.
    3. The observation is partial (the agent does NOT receive its exact location).
    """

    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(
        self,
        render_mode=None,
        map_size=4,          # total map is map_size x map_size
        hole_count=3,        # number of holes
        partial_obs_window=1,# half-size of local window around the agent
        bear_chance=0.25,    # probability that polar bear appears when stepping on a hole
        negative_reward=-1.0,# negative reward if the bear catches the agent
        goal_reward=1.0      # reward if the agent reaches goal
    ):
        super().__init__()
        self.map_size = map_size
        self.hole_count = hole_count
        self.bear_chance = bear_chance
        self.negative_reward = negative_reward
        self.goal_reward = goal_reward

        self.partial_obs_window = partial_obs_window
        self.render_mode = render_mode

        # We define the discrete action space: 4 directions
        # 0: up, 1: right, 2: down, 3: left
        self.action_space = spaces.Discrete(4)

        # For partial observation, let's define a small 2D window around the agent.
        # e.g., partial_obs_window=1 => a 3x3 window (the agent is in the center).
        obs_h = self.partial_obs_window * 2 + 1
        obs_w = self.partial_obs_window * 2 + 1

        # We can encode each cell in that window as an integer in [0..4]:
        #   0 = safe floor
        #   1 = hole
        #   2 = goal
        #   3 = polar bear
        #   4 = agent
        #
        # So the observation shape is (obs_h, obs_w). If you want to keep the
        # original code's "discrete state" approach, you'll need a wrapper
        # to convert this grid into a single int. But let's keep it as a
        # simple Box for demonstration, then we can flatten it ourselves.
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(obs_h, obs_w), dtype=np.int32
        )

        # Internal states
        self.agent_pos = None
        self.holes = []
        self.goal_pos = None
        self.bear_pos = None
        self.bear_active = False

        # Prepare pygame for rendering if needed
        if self.render_mode == "human":
            pygame.init()
            self.window_size = 512
            self.cell_size = self.window_size // self.map_size
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            
            # Load polar bear image (please replace "polar_bear.png" with your actual file).
            img_path = os.path.join(os.path.dirname(__file__), "polar_bear.png")
            if os.path.exists(img_path):
                self.bear_image = pygame.image.load(img_path)
                self.bear_image = pygame.transform.scale(self.bear_image, (self.cell_size, self.cell_size))
            else:
                self.bear_image = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 1) Randomly place holes
        all_positions = [(r, c) for r in range(self.map_size) for c in range(self.map_size)]
        # Choose goal location (e.g. bottom-right)
        self.goal_pos = (self.map_size - 1, self.map_size - 1)
        # Remove the goal from possible holes
        available_positions = [p for p in all_positions if p != self.goal_pos]
        # Randomly select holes
        self.holes = random.sample(available_positions, self.hole_count)

        # 2) Agent starts top-left or somewhere else
        self.agent_pos = (0, 0)

        # 3) Bear is inactive initially
        self.bear_pos = None
        self.bear_active = False

        observation = self._get_observation()
        return observation, {}  # gymnasium requires returning (obs, info)

    def step(self, action):
        # Move the agent
        r, c = self.agent_pos
        if action == 0:    # up
            r = max(r - 1, 0)
        elif action == 1:  # right
            c = min(c + 1, self.map_size - 1)
        elif action == 2:  # down
            r = min(r + 1, self.map_size - 1)
        elif action == 3:  # left
            c = max(c - 1, 0)
        self.agent_pos = (r, c)

        reward = 0.0
        terminated = False
        truncated = False

        # Check if agent stepped on a hole => 25% chance to spawn bear
        if self.agent_pos in self.holes and not self.bear_active:
            if random.random() < self.bear_chance:
                self.bear_active = True
                self.bear_pos = random.choice(self.holes)

        # If bear is active, move it to chase the agent
        if self.bear_active and self.bear_pos is not None:
            self._move_bear_toward_agent()
            # Check if bear catches the agent
            if self.bear_pos == self.agent_pos:
                reward = self.negative_reward
                terminated = True

        # Check if agent reached goal
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            terminated = True

        observation = self._get_observation()
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((0, 0, 0))  # black background

        # Draw tiles
        for r in range(self.map_size):
            for c in range(self.map_size):
                rect = pygame.Rect(c*self.cell_size, r*self.cell_size, self.cell_size, self.cell_size)
                
                # Hole or safe floor?
                if (r, c) in self.holes:
                    color = (60, 60, 60)  # holes
                else:
                    color = (100, 180, 100)  # safe floor
                pygame.draw.rect(self.screen, color, rect)

        # Draw goal
        gr, gc = self.goal_pos
        rect = pygame.Rect(gc*self.cell_size, gr*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 0, 255), rect)

        # Draw agent
        ar, ac = self.agent_pos
        rect = pygame.Rect(ac*self.cell_size, ar*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # Draw bear
        if self.bear_active and self.bear_pos is not None:
            br, bc = self.bear_pos
            if self.bear_image is not None:
                self.screen.blit(self.bear_image, (bc*self.cell_size, br*self.cell_size))
            else:
                # draw a white square if no image is found
                rect = pygame.Rect(bc*self.cell_size, br*self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (255,255,255), rect)

        pygame.display.flip()
        pygame.time.Clock().tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def _get_observation(self):
        """
        Return a partial observation around the agent.
        The shape is (partial_obs_window*2+1, partial_obs_window*2+1).
        Values: 0=safe floor, 1=hole, 2=goal, 3=bear, 4=agent
        """
        obs_size = self.partial_obs_window * 2 + 1
        center_r, center_c = self.agent_pos
        grid = np.zeros((obs_size, obs_size), dtype=np.int32)

        for i in range(obs_size):
            for j in range(obs_size):
                r = center_r - self.partial_obs_window + i
                c = center_c - self.partial_obs_window + j

                if r < 0 or r >= self.map_size or c < 0 or c >= self.map_size:
                    val = 0  # outside bounds => treat as floor
                else:
                    if (r, c) == self.agent_pos:
                        val = 4
                    elif (r, c) == self.goal_pos:
                        val = 2
                    elif self.bear_active and (r, c) == self.bear_pos:
                        val = 3
                    elif (r, c) in self.holes:
                        val = 1
                    else:
                        val = 0
                grid[i, j] = val

        return grid

    def _move_bear_toward_agent(self):
        """Simple chase: bear moves one step closer (Manhattan) to the agent."""
        if not self.bear_active or self.bear_pos is None:
            return
        br, bc = self.bear_pos
        ar, ac = self.agent_pos
        dr = ar - br
        dc = ac - bc
        # Move whichever axis is further first
        if abs(dr) > abs(dc):
            br += np.sign(dr)
        else:
            bc += np.sign(dc)
        # stay within bounds
        br = np.clip(br, 0, self.map_size-1)
        bc = np.clip(bc, 0, self.map_size-1)
        self.bear_pos = (br, bc)
