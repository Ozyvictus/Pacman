Maze / Pac-Man RL Agent (DQN)

This project is a simple reinforcement learning implementation where an agent learns to navigate a maze, avoid obstacles and a moving enemy, and reach a goal using a neural network.

The idea is to build an agent that can learn on its own instead of following hardcoded rules.

At the beginning, the agent moves randomly. Over time, it starts learning:
--> which paths lead to the goal
--> how to avoid walls
--> how to stay away from the monster

This is done using a Deep Q-Network (DQN).

Each step follows this loop:

--> Agent observes the current state
--> Picks an action (random or predicted)
--> Environment updates
--> Agent gets a reward
--> Experience is stored
--> Model is trained

This repeats for many episodes until the agent improves.

Each state includes:

--> Direction to the goal
--> Direction to the monster
--> Distance to goal
--> Nearby walls (3×3 grid)

I also stack the last 3 states, so the model gets 42 inputs total.

Simple feedforward neural network:

Input: 42 features
Hidden layers: 64 → 32
Output: 4 actions (Up, Down, Left, Right)
