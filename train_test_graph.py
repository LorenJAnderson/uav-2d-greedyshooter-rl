# plots training and testing dynamics

import os
import tensorflow as tf
import matplotlib.pyplot as plt

# path to TensorFlow summary log
PATH = "runs/Oct30_16-01-43_loren-XPS-15-9560-DQN/events.out.tfevents.1572469304.loren-XPS-15-9560.12547.0"

train_rewards = []
test_rewards = []

for event in tf.compat.v1.train.summary_iterator(PATH):
    for value in event.summary.value:
        if value.tag == 'reward_100':
            train_rewards.append(value.simple_value)
        elif value.tag == 'test_reward':
            test_rewards.append(value.simple_value)

cwd = os.getcwd()
dir = os.path.join(cwd, "graph")
if not os.path.exists(dir):
    os.mkdir(dir)

plt.plot(range(52, 3952), train_rewards[51::], label="Train", color='b')  # first few noisy rewards are removed
plt.plot(range(701, 3952, 650), test_rewards, label="Test", color='r')  # x-axis scaled to align with first plot
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel("Games", fontsize=16)
plt.ylabel("Reward", fontsize=16)
plt.title("Performance of RL Agent", fontsize=18)
ax = plt.gca()
ax.tick_params(axis="both", which="major", labelsize=16)
ax.tick_params(axis="both", which="minor", labelsize=16)
plt.legend()
plt.savefig("graph/train_test_plot.pdf")
plt.close()
