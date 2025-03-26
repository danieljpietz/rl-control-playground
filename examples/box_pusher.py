from rl_utils.nn import MLP
from rl_utils.ddpg import Agent
from rl_utils import env

import numpy as np
from matplotlib import pyplot as plt

import gym


def simulate(agent, env_name):
    env = gym.make(env_name)
    state = env.reset()
    if isinstance(state, tuple):  # Gym >= 0.26
        state = state[0]

    states = []
    actions = []
    rewards = []

    for _ in range(200):
        action = agent.select_action(state, explore=False)
        next_state, reward, done, *_ = env.step(action)
        if isinstance(next_state, tuple):  # Gym >= 0.26
            next_state = next_state[0]

        actions.append(action)
        states.append(state)
        rewards.append(reward)
        state = next_state

        if done:
            break

    env.close()

    # Plotting
    states = np.array(states)
    x = states[:, 0]

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(x, label="pos (m)")
    plt.title("Position")
    plt.xlabel("Timestep")
    plt.ylabel("pos(m)")
    plt.ylim((-10, 10))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(actions, label="Force")
    plt.title("Force (N)")
    plt.xlabel("Timestep")
    plt.ylabel("Force (N)")
    plt.ylim((-2, 2))
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # env = gym.make("Pendulum-v1")
    env = gym.make("BoxPusher-v0")

    observation_space = env.observation_space.shape[-1]
    action_space = env.action_space.shape[-1]
    action_scale = env.action_space.high[-1]

    controller = MLP(
        (observation_space, 400, 400, 300, action_space), normalize=action_scale
    )
    critic = MLP((observation_space + action_space, 400, 400, 300, 1))

    agent = Agent(env, controller, critic)

    episodes = 200
    for ep in range(episodes):
        state, *_ = env.reset()
        agent.noise.reset()
        episode_reward = 0

        for t in range(200):
            action = agent.select_action(state)
            next_state, reward, done, *_ = env.step(action)
            agent.replay_buffer.add((state, action, reward, next_state, float(done)))
            agent.train()

            state = next_state
            episode_reward += reward
            if done:
                break

        print(f"Episode {ep + 1}, Reward: {episode_reward:.2f}")
        if ep % 10 == 0:
            simulate(agent, "BoxPusher-v0")


if __name__ == "__main__":
    main()

