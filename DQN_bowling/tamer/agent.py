from tamer.interface import Interface
import pygame
import pickle
from itertools import count
from pathlib import Path
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch import nn

torch.manual_seed(10)
ACTION_MAP = {0: 'Noop', 1: 'Fire', 2: 'Up', 3: 'Down'}
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
loss_list = []

'''
----------Encoder Head ========>> Q Function Reward model-----------
'''


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)

        self.conv_bn1 = nn.BatchNorm2d(64)  # channels
        self.conv_bn2 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = torch.relu(self.conv_bn1(self.conv1(x)))  # 160*160
        x = F.max_pool2d(x, 2)  # 80*80
        x = torch.relu(self.conv_bn1(self.conv2(x)))  # 80*80
        x = F.max_pool2d(x, 2)  # 40*40
        x = torch.relu(self.conv_bn1(self.conv2(x)))  # 40*40
        x = F.max_pool2d(x, 2)  # 20*20
        x = torch.relu(self.conv_bn2(self.conv3(x)))  # 10*10
        x = F.max_pool2d(x, 2)  # 10*10
        x = x.view(x.size(0), -1)
        return x


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.linear_1 = nn.Linear(100, 16)
        self.linear_2 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class BufferDeque:
    def __init__(self, size):
        self.memory1 = deque(maxlen=size)
        self.memory2 = deque(maxlen=500)

    def __len__(self):
        return len(self.memory1)

    def push1(self, sample):
        self.memory1.append(sample)

    def push2(self, sample):
        self.memory2.append(sample)


def random_sample(buffer, batch_size):
    rand_idx = np.random.randint(len(buffer), size=batch_size)
    rand_batch = [buffer[i] for i in rand_idx]
    state, action, reward, nxt_state_ls, done_ls = [], [], [], [], []
    for s, a, f, nxt_state, done in rand_batch:
        state.append(s)
        action.append(a)
        reward.append(f)
        nxt_state_ls.append(nxt_state)
        done_ls.append(done)
    return torch.cat(state), torch.tensor(action), torch.tensor(reward), torch.cat(nxt_state_ls), done_ls


class FunctionApproximation:
    def __init__(self, env, encoder, head, batch):
        self.env = env
        self.encoder = encoder
        self.head = head
        self.img_dims = (3, 160, 160)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.98
        self.opt = optim.Adam(list(self.head.parameters()), lr=1e-3)
        self.batch = batch

    def predict(self, f):
        output = self.head(f.to(self.device))
        return output.cpu()

    def update1(self, f, action, reward, nxt_f, done_ls):
        f = f.to(self.device)
        reward = reward.to(self.device)
        action = action.to(self.device)
        h_hat = self.head(f)
        h_hat_s_a = h_hat.gather(1, action.unsqueeze(0).t()).t()[0]
        H_hat_s_a = self.head(nxt_f).max(dim=1)[0]
        y = []
        for _ in range(self.batch):
            if done_ls[_]:
                y.append(reward[_])
            else:
                y.append(H_hat_s_a[_] * self.gamma + reward[_])
        y = torch.stack(y)
        y = y.detach()
        self.opt.zero_grad()
        loss = torch.mean((h_hat_s_a - y) ** 2)
        loss.backward()
        self.opt.step()
        loss_list.append(loss.cpu().item())
        return loss

    def transfer(self, state):
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        state = torch.from_numpy(state)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((self.img_dims[1:])),
                            T.ToTensor()])
        state = resize(state).to(self.device).unsqueeze(0)
        return state


def reward_judge(reward):
    if reward != 0:
        if reward < 10:
            return 2
        else:
            return 3
    else:
        return 0


class DeepTamer:
    def __init__(
            self,
            env,
            encoder,
            head,
            num_episodes,
            batch,
            ts_len=0.3,
            epsilon=0.9,
            min_eps=0,
    ):
        self.env = env
        self.ts_len = ts_len
        self.num_episodes = num_episodes
        self.episode = 0
        self.batch = batch
        self.H = FunctionApproximation(env, encoder, head, self.batch)
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.buffer = BufferDeque(20000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sliding_window = deque()
        self.t = 0
        self.reward_ls = []
        self.video_size = None
        self.epsilon_step = 2e-5

    def act(self, f, train, idx):
        random = True if np.random.random()< 0.3 else False
        if train and random:
            return np.random.randint(0, 4) if np.random.random() < self.epsilon\
                else int(torch.argmax(self.H.predict(f)))
        else:
            return int(torch.argmax(self.H.predict(f)))

    def _train_episode(self, idx, disp, train, render=True):
        self.episode += 1
        self.t = 0
        initialized = False
        if train:
            print(f'train Episode: {idx + 1}')
        else:
            print(f'play Episode: {idx + 1}')
        tot_reward = 0
        obs = self.env.reset()
        temp_buffer = []
        for _ in count():
            obs = obs.transpose((2, 0, 1))
            state = self.H.transfer(obs)
            with torch.no_grad():
                f = self.H.encoder(state)
            action = self.act(f, train, idx)
            if not initialized and not train:
                action = 1
                initialized = True
            nxt_obs, reward, done, info = self.env.step(action)
            if done:
                break
            tot_reward += reward
            nxt_state = self.H.transfer(nxt_obs.transpose((2, 0, 1)))
            with torch.no_grad():
                nxt_f = self.H.encoder(nxt_state)
            data = [f, action, self.t, nxt_f, done]
            if train:
                self.buffer.push2(data)
                if reward_judge(reward):
                    for data in self.buffer.memory2:
                        for i in range(reward_judge(reward)):
                            if self.t - 40 - i * 135 > data[2] > self.t - (40 + 95) - i * 135:
                                data[2] = reward
                                self.buffer.push1(data)
                                temp_buffer.append(data)
                    self.buffer.memory2.clear()
                    for _ in range(100):
                        f_, action_, reward_, nxt_f_, done_ls = random_sample(temp_buffer, self.batch)
                        self.H.update1(f_, action_, reward_, nxt_f_, done_ls)
                    temp_buffer = []
                    self.t = 0
            if self.t > 400 and reward == 0 and train:
                data = [f, action, -10, nxt_f, done]
                self.buffer.push1(data)
            if render:
                self.env.render()
            disp.show_action(action)
            if train:
                if len(self.buffer) > 40 and self.t % 5 == 0:
                    for i in range(20):
                        state_, action_, reward_, nxt_state_, done_ls = random_sample(self.buffer.memory1, self.batch)
                        loss = self.H.update1(state_, action_, reward_, nxt_state_, done_ls)
                    if self.t % 250 == 0:
                        print(loss)
            self.t += 1
            if done:
                break
                tot_reward += reward
            obs = nxt_obs
            if self.epsilon > self.min_eps:
                self.epsilon = self.epsilon-self.epsilon_step
        self.env.close()
        return tot_reward

    def train(self, model_file_to_save=None):
        self.env.render()
        disp = Interface(action_map=ACTION_MAP)

        for i in range(self.num_episodes):
            self._train_episode(i, disp, render=True, train=True)
            print('offline')
            self.play(num=0)

        print('\nCleaning up...')
        self.env.close()

        if model_file_to_save is not None:
            self.save_model(filename=model_file_to_save)

        plt.title('Head_Network_Error')
        plt.plot(loss_list)
        plt.savefig('Test_Error')
        print(self.reward_ls)
        np.save('reward_ls.npy', self.reward_ls)
        print('Finished Successfully!')
        return

    def save_model(self, filename):
        model = self.H
        filename = filename + '.p'
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)
        return

    def update_screen(self, screen, arr):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        pyg_img = pygame.transform.scale(pyg_img, self.video_size)
        screen.blit(pyg_img, (0, 0))

    def play(self, num):
        print("play")
        disp = Interface(action_map=ACTION_MAP)
        play_reward_ls = []
        temp = self._train_episode(-1, disp=disp, render=True, train=False)
        for i in range(num):
            play_reward_ls.append(self._train_episode(i, disp=disp, render=False, train=False))
        Sum = 0
        for reward in play_reward_ls:
            Sum = Sum + reward
        mean = temp
        print(f'score:{mean}')
        self.reward_ls.append(mean)
        self.env.close()
