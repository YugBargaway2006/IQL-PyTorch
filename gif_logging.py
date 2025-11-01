import gymnasium as gym
import imageio
import numpy as np


def main():
    env = gym.make("<ENV_NAME", render_mode="rgb_array")

    random_policy = env.action_space.sample()
    evaluate_policy(env, random_policy)


def evaluate_policy(env, policy, filename="eval_policy.gif", episodes=1, fps=30):
    frames = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            frame = env.render(mode="rgb_array")
            frames.append(frame)

            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved recorded policy as {filename}")


main()