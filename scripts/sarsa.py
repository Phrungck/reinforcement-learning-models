import numpy as np


def sarsa(env, episode=max_ep, step_count=max_step, discount=gamma, alpha=alpha, epsilon=eps):

    # average array
    arr = []

    # sarsa rewards array
    s_rewards_arr = []

    # Sarsa states and action array
    SRS = np.zeros((env.observation_space.n, env.action_space.n))

    # epsilon function
    def epsilonPolicy(s, epsilon):
        p = np.random.rand()
        if p < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(SRS[s])
        action = int(action)
        return action

    # cumulative reward variable
    total_reward = 0

    for i in range(episode):
        s = env.reset()

        # current action in current policy
        a = epsilonPolicy(s, epsilon)

        # rewards per episode
        episode_reward = 0

        for j in range(step_count):

            s_, reward, done, info = env.step(a)

            # next action
            a_ = epsilonPolicy(s_, epsilon)

            # TD Update
            predict = SRS[s, a]
            target = reward + discount*SRS[s_, a_]
            SRS[s, a] = predict + alpha*(target-predict)

            # storing rewards for episode
            episode_reward = episode_reward + reward
            total_reward = total_reward + reward

            if done:
                break

            s = s_
            a = a_

        # storing the average rewards
        arr = np.append(arr, episode_reward)
        if i % 40 == 0:
            average = np.average(arr[i-40:i])
            new_count = i/40

        epsilon = max(min_eps, np.exp(-decay*i))
        s_rewards_arr = np.append(s_rewards_arr, np.array(
            [episode_reward, total_reward, i, average, new_count]))

    return SRS, s_rewards_arr.reshape(-1, 5)
