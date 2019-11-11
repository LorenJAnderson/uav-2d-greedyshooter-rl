import argparse
import os
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import uav_environment


GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000
TEST_FRAMES = 25000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


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


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = np.random.randint(10)
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.FloatTensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        angle = ((action % 5) - 2) * np.pi/24
        vel = 0.05 if action < 5 else 0.1

        new_state, reward, is_done, _ = self.env.step(angle, vel)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def test_agent(net):
    test_env = uav_environment.Environment(vel=0.1, steps=100, turn=np.pi/12,
                                           cap_dist=0.25, cap_angle=np.pi/24, start_dev=0.5)
    cumulative_reward = 0
    for greedy_angle in range(2):
        for uav_angle in range(0, 8):
            for distance in np.linspace(0.5, 0.9, 5):
                init_uav_angle = uav_angle * np.pi / 4
                init_uav_state = np.array([distance, 0, init_uav_angle])
                init_greedy_angle = 0 if greedy_angle == 0 else np.pi
                init_greedy_state = np.array([0, 0, init_greedy_angle])
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
    return cumulative_reward/80


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    cwd = os.getcwd()
    dir = os.path.join(cwd, "test")
    if not os.path.exists(dir):
        os.mkdir(dir)

    env = uav_environment.Environment(vel=0.1, steps=100, turn=np.pi/12,
                                      cap_dist=0.25, cap_angle=np.pi/24, start_dev=0.5)

    net = DQN(3, 10, 64).to(device)
    tgt_net = DQN(3, 10, 64).to(device)
    writer = SummaryWriter(comment="-DQN")

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_rewards = []
    frame_idx = 0
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f" % (
                frame_idx, len(total_rewards), mean_reward, epsilon
            ))
            writer.add_scalar("reward_100", mean_reward, frame_idx)

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())
        if frame_idx % TEST_FRAMES == 0:
            test_reward = test_agent(net)
            torch.save(net.state_dict(), "test/" + str(round(test_reward, 3)) + str(frame_idx) + "-" + "-DQN-test.dat")
            writer.add_scalar("test_reward", test_reward, frame_idx)
            if test_reward == 1:
                break

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()
