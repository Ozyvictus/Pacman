import pygame
import numpy as np
import agent
from environment import MazeEnv, init_pygame, render
from agent import Agent
import matplotlib.pyplot as plt

EPISODES = 300


def main():
    env = MazeEnv(size=10)
    agent = Agent(state_size=42, action_size=4)

    screen = init_pygame(env.size)
    scores = []
    mean_scores = []
    total_score = 0

    for episode in range(EPISODES):
        state = env.reset()
        agent.reset_memory()
        state = agent.get_stacked_state(state)
        total_reward = 0

        while True:
            # Handle quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.act(state)
            next_state_raw, reward, done = env.step(action)
            next_state = agent.get_stacked_state(next_state_raw)

            # Training
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)
            agent.train_long_memory()

            state = next_state
            total_reward += reward

            render(screen, env)
            pygame.time.delay(50)

            if done:
                break

        agent.decay_epsilon()

        scores.append(total_reward)
        total_score += total_reward

        mean_score = total_score / (episode + 1)
        mean_scores.append(mean_score)

        print(f"Episode {episode} | Score: {total_reward:.2f} | Mean: {mean_score:.2f} | Epsilon: {agent.epsilon:.3f}")

    pygame.quit()
    
    plt.figure(figsize=(10,5))
    plt.plot(scores, label="Score per Episode")
    plt.plot(mean_scores, label="Average Score", linewidth=3)

    plt.xlabel("Episodes")  
    plt.ylabel("Score")
    plt.title("Learning Curve")

    plt.legend()
    plt.grid()

    plt.savefig("training_plot.png")  # saved for report
    plt.show()
    
    

if __name__ == "__main__":
    main()