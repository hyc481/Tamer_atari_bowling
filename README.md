# Tamer_atari_bowling
This is part of my graduate design for my undergraduate study. The main task is to carry out a series of experiments to look into the influence of introducing 
humans into training of reinforcement learning agents. This method is also called human-in-the-loop rl or interactive rl. PLease refer to [this site](https://arxiv.org/pdf/1709.10163.pdf) 
for the original paper and deep tamer framework. Apart from the implementation of deep tamer framework, PPO method and DQN method with a modified reward mechanism 
are also experimented. All these three methods achieve good scores. To improve the agent's performance, an encoder is adopted in all these three methods. Please refer to the original paper for more detailed information.

[Repository 1](https://github.com/bharadwaj1098/Tamer) and [Repository 2](https://github.com/benibienz/TAMER) provided great help in coding when I was working on
the project.

Below are some brief results of my experiments. I chose the classic DQN method as comparison.

__The tamer framework__

<div align=center><img src="https://github.com/hyc481/Tamer_atari_bowling/assets/141563901/a472cee9-7477-437a-ad70-3928f1999689" width="450" height="300"><br/>
</div>

__The DQN method with a modified reward mechanism__


<div align=center><img src="https://github.com/hyc481/Tamer_atari_bowling/assets/141563901/e1c24352-ea76-44a7-b901-499601e9c380" width="450" height="300"><br/>
</div>

The reward mechanism in the original game can cause quite a great challenge for DQN method, which is extremely sparse. So I modified the mechanism
to make it less sparse and deleted some undesired image frames in the memory pool. The method is crude but the improvement is remarkable.

__The PPO method__

<div align=center><img src="https://github.com/hyc481/Tamer_atari_bowling/assets/141563901/c98f77c5-abdc-45f5-ab33-c60a010c8a63" width="450" height="300"><br/>
</div>

As a baseline algorithm, PPO algorithm is experimented as well. Only a single agent is used to collect data.
