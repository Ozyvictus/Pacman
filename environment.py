import numpy as np
import random
import pygame

class MazeEnv:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.monster_pos = [self.size//2, self.size//2]

        # Create grid (0 = empty, 1 = wall)
        self.grid = np.zeros((self.size, self.size))

        # Add walls (simple maze)
        self.grid[2, 2:7] = 1
        self.grid[5, 1:5] = 1
        self.grid[7, 3:9] = 1

        self.done = False
        return self.get_state()

    def step(self, action):
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        move = moves[action]

        reward = -0.1  # step penalty

        # Calculate new position
        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]

        # Check valid move
        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if self.grid[new_x][new_y] == 0:
                self.agent_pos = [new_x, new_y]
            else:
                reward = -5  # hit wall
        else:
            reward = -5  # boundary hit

        # Goal
        if self.agent_pos == self.goal_pos:
            reward = 10
            self.done = True

        # Monster
        if self.agent_pos == self.monster_pos:
            reward = -10
            self.done = True

        # Move monster
        self.move_monster()

        return self.get_state(), reward, self.done

    def move_monster(self):
        moves = [(-1,0),(1,0),(0,-1),(0,1)]
        move = random.choice(moves)

        new_x = self.monster_pos[0] + move[0]
        new_y = self.monster_pos[1] + move[1]

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            if self.grid[new_x][new_y] == 0:
                self.monster_pos = [new_x, new_y]

    # 🔥 LOCAL VIEW (3x3 around agent)
    def get_local_view(self):
        view = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                x = self.agent_pos[0] + dx
                y = self.agent_pos[1] + dy

                if 0 <= x < self.size and 0 <= y < self.size:
                    view.append(self.grid[x][y])
                else:
                    view.append(1)  # treat out-of-bounds as wall
        return view

    # 🔥 IMPROVED STATE
    def get_state(self):
        dx_goal = (self.goal_pos[0] - self.agent_pos[0]) / self.size
        dy_goal = (self.goal_pos[1] - self.agent_pos[1]) / self.size

        dx_monster = (self.monster_pos[0] - self.agent_pos[0]) / self.size
        dy_monster = (self.monster_pos[1] - self.agent_pos[1]) / self.size

        distance = np.sqrt(dx_goal**2 + dy_goal**2)

        local_view = self.get_local_view()

        return np.array([
            dx_goal, dy_goal,
            dx_monster, dy_monster,
            distance,
            *local_view
        ], dtype=float)


# ================= VISUALS =================

CELL_SIZE = 40
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (100,100,100)
YELLOW = (255,255,0)
RED = (255,0,0)
GREEN = (0,255,0)

def init_pygame(size):
    pygame.init()
    screen = pygame.display.set_mode((size*CELL_SIZE, size*CELL_SIZE))
    pygame.display.set_caption("Pac-Man RL Agent")
    return screen

def render(screen, env):
    screen.fill(BLACK)

    # Draw grid + walls
    for x in range(env.size):
        for y in range(env.size):
            rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if env.grid[x][y] == 1:
                pygame.draw.rect(screen, GRAY, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect, 1)

    # Draw goal (food)
    pygame.draw.circle(
        screen, GREEN,
        (env.goal_pos[1]*CELL_SIZE + CELL_SIZE//2,
         env.goal_pos[0]*CELL_SIZE + CELL_SIZE//2),
        6
    )

    # Draw agent (Pac-Man)
    pygame.draw.circle(
        screen, YELLOW,
        (env.agent_pos[1]*CELL_SIZE + CELL_SIZE//2,
         env.agent_pos[0]*CELL_SIZE + CELL_SIZE//2),
        CELL_SIZE//2 - 4
    )

    # Draw monster (ghost)
    pygame.draw.circle(
        screen, RED,
        (env.monster_pos[1]*CELL_SIZE + CELL_SIZE//2,
         env.monster_pos[0]*CELL_SIZE + CELL_SIZE//2),
        CELL_SIZE//2 - 4
    )

    pygame.display.flip()