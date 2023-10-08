"""
When training, use 'A' and 'D' keys for positive and negative rewards
"""
import gym
import asyncio
import torch
from tamer.agent import Agent, Encoder


async def main():
    env = gym.make("Bowling-v0", mode=2).unwrapped
    env = gym.wrappers.TimeLimit(env, max_episode_steps=4000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("auto_encoder/Type_1/encoder.pt", map_location=device))

    for name, params in encoder.named_parameters():
        params.requires_grad = False

    encoder = encoder.eval()

    num_episodes = 50
    epochs = 40
    lmbda = 0.9
    eps = 0.075
    gamma = 0.98
    agent = Agent(env=env, encoder=encoder, num_episodes=num_episodes,
                  gamma=gamma, epochs=epochs, lmbda=lmbda, eps=eps)

    agent.train(model_file_to_save='auto_save')

if __name__ == '__main__':
    asyncio.run(main())




