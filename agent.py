import numpy as np
import random


def getReward(ball, bar):
    if bar.top <= ball.centery <= bar.bottom:
        return 1
    else:
        return -1


def get_state_from_pos(y, bar_height, window_height):
    a = 0
    b = bar_height
    for i in range(0, window_height, bar_height):
        if a <= y <= b:
            return int((b / bar_height) - 1)
        else:
            a += bar_height
            b += bar_height


class Qlearning:

    def __init__(self, alpha, gamma, eps, window_height, bar_height, eps_decay=0.):
        self.states = int(window_height / bar_height)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        # Q-Table
        self.qTable = np.matrix(np.zeros(shape=[self.states, self.states]))
        self.rewards = []

    def get_action(self, s):
        if s < 0:
            s = 0
        if s >= self.states:
            s = self.states - 1
        a = np.argmax(self.qTable[s, :])

        if a < 0:
            a = 0
        if a > self.states:
            a = self.states
        return a

    def update(self, s, bar, ball):
        global s_
        if s < 0:
            s = 0
        if s >= self.states:
            s = self.states - 1

        exp_threshold = random.uniform(0, 1)
        # if exploiting
        if exp_threshold > self.eps:
            a = self.get_action(s)
        else:  # if exploring
            a = random.choice([i for i in range(self.states)])

        if a >= 0:
            s_ = a

        reward = getReward(ball, bar)
        self.rewards.append(reward)

        self.qTable[s, a] += self.alpha * (reward + self.gamma * np.max(self.qTable[s_, :]) - self.qTable[s, a])

        return s_ * bar.height
