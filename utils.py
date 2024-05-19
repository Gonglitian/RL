import torch
import random
# 数据池


class Pool:

    def __init__(self, controller: 'Controller'):
        self.pool = []
        self.controller = controller

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, i):
        return self.pool[i]

    # 更新动作池
    def update(self):
        # 每次更新不少于N条新数据
        old_len = len(self.pool)
        len_new_data = 0
        while len(self.pool) - old_len < 10_000:
            new_data = self.controller.play()[0]
            len_new_data += len(new_data)
            # print(len_new_data)
            self.pool.extend(new_data)

        # 只保留最新的N条数据
        self.pool = self.pool[-100_000:]
        return len_new_data

    def sample(self):
        data = random.sample(self.pool, 256)

        state = torch.FloatTensor(
            [i[0] for i in data]).reshape(-1, self.controller.observation_length)
        action = torch.LongTensor([i[1] for i in data]).reshape(-1, 1)
        reward = torch.FloatTensor([i[2] for i in data]).reshape(-1, 1)
        next_state = torch.FloatTensor(
            [i[3] for i in data]).reshape(-1, self.controller.observation_length)
        terminated = torch.LongTensor([i[4] for i in data]).reshape(-1, 1)

        return state, action, reward, next_state, terminated


class Controller():
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.observation_length = env.observation_space.shape[0]
        self.action_length = env.action_space.n

    def play(self, mode="train", show=False):
        data = []
        reward_sum = 0

        s, info = self.env.reset()
        terminated = False
        truncated = False
        while not terminated:
            a = self.model(torch.FloatTensor(s).reshape(
                1, self.observation_length)).argmax().item()
            if mode == "train" and random.random() < 0.1:
                a = self.env.action_space.sample()

            ns, r, terminated, truncated, info = self.env.step(a)
            data.append((s, a, r, ns, terminated))

            reward_sum += r
            s = ns
            if truncated:
                # print("truncated")
                break
            if show:
                self.env.render()
        return data, reward_sum
