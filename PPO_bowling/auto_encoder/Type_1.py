import os
import pathlib
import random

import gym
import matplotlib.pyplot as plt
import numpy as np

from itertools import count

import torch
import torchvision
import torchvision.utils
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)

'''
This code is to make auto-encoder train till episode is DONE.
each state is 1 RGB image.
'''


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # in_channel, out_channel, kernel_size, stride=1, padding=0
        # conv: height_out = (height_in - height_kernel + 2*padding) / stride + 1
        # pool: height_out = (height_in - height_kernel) / stride + 1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)

        self.conv_bn1 = nn.BatchNorm2d(64)   # channels
        self.conv_bn2 = nn.BatchNorm2d(1)


    def forward(self, x):
        x = torch.relu(self.conv_bn1(self.conv1(x))) #160*160
        x = F.max_pool2d(x, 2)                       #80*80
        x = torch.relu(self.conv_bn1(self.conv2(x))) #80*80
        x = F.max_pool2d(x, 2)                       #40*40
        x = torch.relu(self.conv_bn1(self.conv2(x))) #40*40
        x = F.max_pool2d(x, 2)                       #20*20
        x = torch.relu(self.conv_bn2(self.conv3(x))) #10*10
        x = F.max_pool2d(x, 2)                       #10*10
        x = x.view(x.size(0), -1)
        return x

class Decoder(nn.Module):
    # Linear - Reshape - Upsample - Deconv + BatchNorm
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv_1 = nn.ConvTranspose2d(1, 64, 3, padding=1)
        self.deconv_2 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.deconv_3 = nn.ConvTranspose2d(64, 3, 3, padding=1)

        self.deconv_bn1 = nn.BatchNorm2d(64)
        self.deconv_bn2 = nn.BatchNorm2d(3)

        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # conv -----> Bn -----> Activate ----> Pool
        # Up-Pool ----> DeConv  ----> BN ----> Activate
        """
        x = torch.relu(self.linear_1(x))
        x = x.view(x.shape[0], 1, 8, 8)
        x = torch.relu(self.deconv_1(self.upsample_1(x)))
        x = self.deconv_bn1(x)
        x = torch.relu(self.deconv_2(self.upsample_1(x)))
        x = self.deconv_bn1(x)
        x = torch.relu(self.deconv_2(self.upsample_2(x)))
        x = self.deconv_bn1(x)
        x = torch.relu(self.deconv_3(self.upsample_1(x)))
        x = self.deconv_bn2(x)
        x = torch.sigmoid(x)     # sigmoid
        return x
        """
        x = x.view(x.shape[0], 1, 10, 10)
        x = torch.relu(self.deconv_bn1(self.deconv_1(x)))
        x = self.upsample_1(x)
        x = torch.relu(self.deconv_bn1(self.deconv_2(x)))
        x = self.upsample_1(x)
        x = torch.relu(self.deconv_bn1(self.deconv_2(x)))
        x = self.upsample_2(x)
        x = torch.relu(self.deconv_bn2(self.deconv_3(x)))
        x = self.upsample_1(x)
        return x

class BufferArray:
    def __init__(self, size):
        self.memory = torch.zeros(size, dtype=torch.float32)
        self._mem_loc = 0
        self.push_count = 0

    def __len__(self):
        return self.push_count

    def push(self, tensor):
        type(self._mem_loc)
        if self._mem_loc == len(self.memory) - 1:
            self._mem_loc = 0
        self.memory[self._mem_loc] = tensor.cpu()
        self._mem_loc += 1
        self.push_count += 1

    def random_sample(self, batch_size):
        rand_batch = np.random.randint(len(self.memory), size=batch_size)
        if len(rand_batch.shape) == 3:
            return torch.stack(self.memory[rand_batch])
        return self.memory[rand_batch]


def main():
    env = gym.make("Bowling-v0").unwrapped
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    n_actions = env.action_space.n  # 6
    no_of_episodes = 30  # 2000
    batch_size = 32
    img_dims = (3, 160, 160)
    buffer_size = (20000,) + img_dims

    def select_action():
        a = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return a

    def get_screen():
        screen = env.render(mode="rgb_array").transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((img_dims[1:])),
                            T.ToTensor()])
        screen = resize(screen).to(device)
        return screen  # Returns TENSOR

    buffer = BufferArray(buffer_size)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_history = []

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4, weight_decay=1e-1)

    def image_visual():
        batch = 3
        plt.figure(figsize=(25, 30))
        i = buffer.random_sample(batch).to(device)
        o = decoder(encoder(i))

        interleaved_shape = (len(i) + len(o),) + i.shape[1:]
        interleaved = torch.empty(interleaved_shape)
        interleaved[0::2] = i
        interleaved[1::2] = o

        img_grid = torchvision.utils.make_grid(interleaved, nrow=2)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.savefig(os.path.join(pathlib.Path().absolute(), 'Auto/result'))
        plt.close()

    def loss_visual(x):
        plt.plot(x)
        plt.xlabel("Step")
        plt.ylabel("MSE Loss")
        plt.title("Loss of Encoder-Decoder During Training")
        plt.savefig(os.path.join(pathlib.Path().absolute(), 'Auto/loss_history'))
        plt.close()

    def optimize():
        state_batch = buffer.random_sample(batch_size)

        state_batch = state_batch.to(device)
        output_batch = decoder(encoder(state_batch))

        # Compute loss
        loss = loss_fn(state_batch, output_batch).cpu()   # todo: ?

        # Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss

    for episode in range(no_of_episodes):
        env.reset()
        state = get_screen()

        done = False
        for step in count():
            action = select_action()
            _, _, done, _ = env.step(action.item())

            if not done:
                next_state = get_screen()
            else:
                next_state = None

            # Store transition
            buffer.push(state)
            # Set next state
            state = next_state

            if buffer.push_count >= 20000:
                if step % 16 == 0:
                    loss = optimize()
                    loss_history.append(loss.cpu().item())
                    print('Episode: {} Step: {} Loss: {:5f}'.format(episode, step, loss))
                    '''
                    print("Episode: {} Step: {} Loss: {:5f} GPU Memory: {} RAM: {:5f} Buffer Size: {}".format(
                                        episode, step, loss, get_gpu_memory_map()[0]/1000, 
                                        psutil.virtual_memory().available /  1024**3, len(buffer)
                                        ))
                    clear_output(wait=True)
                    '''
            # Check if down
            if done:
                break

    image_visual()
    loss_visual(loss_history)


    torch.save(encoder.state_dict(), 'Auto/encoder.pt')
    torch.save(decoder.state_dict(), 'Auto/decoder.pt')


if __name__ == "__main__":
    main()