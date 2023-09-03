from tamer.interface import Interface
import pygame
import pickle
from itertools import count
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from collections import deque


torch.manual_seed(10)
ACTION_MAP = {0: 'Noop', 1: 'Fire', 2: 'Up', 3: 'Down'}
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
loss_list = []

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # in_channel, out_channel, kernel_size, stride=1, padding=0
        # conv: height_out = (height_in - height_kernel + 2*padding) / stride + 1
        # pool: height_out = (height_in - height_kernel) / stride + 1
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


class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens1, n_hiddens2, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens1)
        self.fc2 = nn.Linear(n_hiddens1, n_hiddens2)
        self.fc3 = nn.Linear(n_hiddens2, n_actions)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        return x


class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens1, n_hiddens2):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens1)
        self.fc2 = nn.Linear(n_hiddens1, n_hiddens2)
        self.fc3 = nn.Linear(n_hiddens2, 1)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, size):  # shape:the dimension of input data
        self.mean_ls = deque(maxlen=size)
        self.std_ls = deque(maxlen=size)
        self.len = 0

    def update(self, Mean, Std):
        self.mean_ls.append(Mean)
        self.std_ls.append(Std)
        self.len = len(self.mean_ls)
        return [(torch.tensor(self.mean_ls)).mean(), (torch.tensor(self.std_ls)).mean()]


def batchNormalization(advantageBatch):
    print('temp')


def transfer(state):
    device = torch.device("cuda")
    state = np.ascontiguousarray(state, dtype=np.float32) / 255
    state = torch.from_numpy(state)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((160, 160)),
                        T.ToTensor()])
    state = resize(state).to(device).unsqueeze(0)
    return state


class Agent:
    def __init__(
            self,
            env,
            encoder,
            num_episodes,
            gamma,
            epochs,
            lmbda,
            eps
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Network
        self.encoder = encoder
        self.actor = PolicyNet(n_states=100, n_hiddens1=64, n_hiddens2=16, n_actions=4).to(self.device)
        self.critic = ValueNet(n_states=100, n_hiddens1=64, n_hiddens2=16).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.num_episodes = num_episodes
        self.episode = 0
        self.t = 0
        self.reward_ls = []
        self.video_size = None
        # PPO parameters
        self.gamma = gamma
        self.epochs = epochs
        self.lmbda = lmbda
        self.eps = eps
        self.runningMeanStd = RunningMeanStd(1000)

    def act(self, f):
        probs = self.actor(f)
        action_list = torch.distributions.Categorical(probs)
        action = action_list.sample().item()
        return action



    def _train_episode(self, idx, disp, train, render=True):
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        self.episode += 1
        self.t = 0
        if train:
            print(f'train Episode: {idx + 1}')
        else:
            print(f'play Episode: {idx + 1}')
        tot_reward = 0
        obs = self.env.reset()
        penalty = 0
        for _ in count():
            obs = obs.transpose((2, 0, 1))
            state = transfer(obs)
            f = self.encoder(state)
            action = self.act(f)

            nxt_obs, reward, done, info = self.env.step(action)
            tot_reward += reward
            if done:
                break
            if reward == 0:
                penalty += -0.1
                if penalty < -80:
                    reward = -50
                    penalty = 0
            elif reward < 2:
                reward = -1
                penalty = 0
            else:
                penalty = 0
            nxt_f = self.encoder(transfer(nxt_obs.transpose((2, 0, 1))))
            transition_dict['states'].append(f)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(nxt_f)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            if render:
                self.env.render()
            disp.show_action(action)
            if train and reward != 0:
                self.learn(transition_dict)
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': [],
                }
            self.t += 1
            if done:
                break
            obs = nxt_obs
        self.env.close()
        return tot_reward

    def train(self, model_file_to_save=None):
        self.env.render()
        disp = Interface(action_map=ACTION_MAP)

        for i in range(self.num_episodes):
            self._train_episode(i, disp, render=True, train=True)
            self.play(num=0)
            if model_file_to_save is not None:
                self.save_model(filename=model_file_to_save+str(i))

        print('\nCleaning up...')
        self.env.close()
        """
        plt.title('Head_Network_Error')
        plt.plot(loss_list)
        plt.savefig('Test_Error')
        """
        print(self.reward_ls)
        np.save('reward_ls_5_13.npy', self.reward_ls)
        plt.plot(self.reward_ls)
        plt.show()
        print('Finished Successfully!')
        return

    def save_model(self, filename):
        model = self.actor
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
        return0 = self._train_episode(-1, disp=disp, render=True, train=False)
        for i in range(num):
            play_reward_ls.append(self._train_episode(i, disp=disp, render=False, train=False))
        Sum = 0
        for reward in play_reward_ls:
            Sum = Sum+reward
        if num == 0:
            mean = return0
        else:
            mean = Sum/num
        print(f'score:{mean}')
        self.reward_ls.append(mean)
        self.env.close()

    def learn(self, transition_dict):
        # 提取数据集
        states = torch.cat(transition_dict['states']).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.cat(transition_dict['next_states']).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        next_q_target = self.critic(next_states)
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        td_value = self.critic(states)
        td_delta = td_target - td_value

        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []

        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(float(advantage))
        advantage_list.reverse()

        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
        advantage = (advantage-advantage.mean())/advantage.var()
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = torch.mul(ratio.t(), advantage)
            surr2 = torch.mul(torch.clamp(ratio, 1 - self.eps, 1 + self.eps).t(), advantage)
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            """
            if _%3 == 0:
                print(actor_loss)
                print(critic_loss)
                print('\n')
            """

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


