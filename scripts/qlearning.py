import numpy as np

# For Q Learning


def qLearning(env, episode=max_ep, step_count=max_step, discount=gamma, alpha=alpha, epsilon=eps):
    # average array
    arr = []
    # q rewards array
    q_rewards_arr = []

    # Q-Table
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # epsilon function
    def epsilonPolicy(s, epsilon):
        p = np.random.rand()
        if p < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[s])
        action = int(action)
        return action

    # cumulative reward variable
    total_reward = 0

    for i in range(episode):
        # s is the current state. This resets the environment
        s = env.reset()

        # reward per episode
        episode_reward = 0

        for j in range(step_count):
            # env.render()

            # choose action according to the probability distribution
            a = epsilonPolicy(s, epsilon)

            # s_ is the next state
            s_, reward, done, info = env.step(a)

            # TD update
            predict = Q[s, a]
            target = reward + discount * np.max(Q[s_])
            Q[s, a] = Q[s, a] + alpha * (target - predict)

            # storing rewards for episode
            episode_reward = episode_reward + reward
            total_reward = total_reward + reward

            if done:
                break

            # storing the next state as the current state
            s = s_

        # storing the average rewards
        arr = np.append(arr, episode_reward)
        if i % 40 == 0:
            average = np.average(arr[i-40:i])
            new_count = i/40

        epsilon = max(min_eps, np.exp(-decay*i))
        q_rewards_arr = np.append(q_rewards_arr, np.array(
            [episode_reward, total_reward, i, average, new_count]))

    return Q, q_rewards_arr.reshape(-1, 5)
