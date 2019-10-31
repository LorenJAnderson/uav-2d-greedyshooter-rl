import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import uav_environment


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, hid_size):
        super(DQN, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(input_shape, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, n_actions),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mu(x)


def test_agent(net):
    test_env = uav_environment.Environment(vel=0.1, steps=100, turn=np.pi/12,
                                           cap_dist=0.25, cap_angle=np.pi/24, start_dev=0.5)
    cumulative_reward = 0
    index = 1
    for i in range(2):
        for pos in range(0, 8):
            angle = pos * np.pi/4
            for dist in range(5, 10):
                distance = dist/10
                init_uav_state = np.array([distance, 0, angle])
                greedy_header = 0 if i == 0 else np.pi
                init_greedy_state = np.array([0, 0, greedy_header])
                state = test_env.reset(uav_state=init_uav_state, greedy_state=init_greedy_state)
                done = False
                while not done:
                    state_a = np.array([state], copy=False)
                    state_v = torch.FloatTensor(state_a).to(device)
                    q_vals_v = net(state_v)
                    _, act_v = torch.max(q_vals_v, dim=1)
                    action = int(act_v.item())

                    angle = ((action % 5) - 2) * np.pi / 24
                    vel = 0.05 if action < 5 else 0.1

                    state, reward, done, info = test_env.step(angle, vel)
                    cumulative_reward += reward
                    if done:
                        trajectory_plotter_animated(info, index)
                index += 1
    return cumulative_reward/80


def trajectory_plotter_animated(history, index):
    for i in range(len(history)):
        state = history[i][0]
        plt.scatter(state[0], state[1], marker=(3, 0, 180/np.pi*state[2]-90), color='blue', label="RL")
        plt.scatter(state[3], state[4], marker=(3, 0, 180/np.pi*state[5]-90), color='red', label="GREEDY")
    plt.savefig("pics/" + str(index) + "-plot.jpg")
    plt.close()


cwd = os.getcwd()
dir = os.path.join(cwd, "pics")
if not os.path.exists(dir):
    os.mkdir(dir)

device = "cuda"
path = "test/150000-1.0-DQN-test.dat"  # path to pytorch model
net = DQN(3, 10, 64).to(device)
net.load_state_dict(torch.load(path))
test_agent(net)




