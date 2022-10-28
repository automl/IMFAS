from torch import nn


class Network(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class DQNAgent:
    def __init__(self, policy_net, target_net):
        self.policy_net = policy_net
        self.target_net = target_net

    def act(self, state):
        pass

    def update(self, state, action, reward, next_state):
        pass
