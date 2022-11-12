import random
from collections import deque, namedtuple
from copy import deepcopy

import math
import torch
from torch import nn, optim
from torch.nn import HuberLoss

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Replaybuffer:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, policy_net, target_net, eps_start, eps_end, eps_decay, device: str = "cpu"):
        super(DQNAgent, self).__init__()
        # fwd will produce Q(s, a) for all a; i.e. predict expected return of
        # an action given a state
        self.policy_net = policy_net
        self.target_net = target_net.load_state_dict(policy_net.state_dict())

        self.steps_done = 0

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.device = device

    def act(self, state):
        # epsilon-greedy
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(7)]],  # n_actions
                                device=self.device,
                                dtype=torch.long)


class Environment:
    def __init__(self):
        self.done = False
        self.cumulative_reward = 0

        # state: 4 wins board
        self.board = torch.zeros(7, 6)

    def __repr__(self):
        return str(self.board)

    @property
    def state(self):
        if self.done:
            return None  # for termination mask in replay!
        return torch.cat([self.board.view(1, -1), self.board.sum()])  # add current cost.

    def update_state(self, action: int):
        column_count = self.board.sum(axis=1)
        self.board[6 - column_count[action], action] = 1

    def reward(self):
        reward = -1  # for every step
        if torch.any(self.board.sum(axis=1) == 4):
            reward = 1
            self.done = True

        self.cumulative_reward += reward

        return reward

    def reset(self):
        self.board = torch.zeros(7, 6)
        self.done = False
        self.cumulative_reward = 0


if __name__ == '__main__':
    env = Environment()

    replay = Replaybuffer(1000)

    policy_net = nn.Sequential(
        nn.Linear(42 + 1, 100),  # brettgröße 7 * 6, +1 for current cost
        nn.ReLU(),
        nn.Linear(100, 7),  # 7 mögliche züge
    )

    target_net = nn.Sequential(
        nn.Linear(42 + 1, 100),
        nn.ReLU(),
        nn.Linear(100, 7),
    )

    agent = DQNAgent(policy_net, target_net)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    lossfn = HuberLoss()

    N_EPISODES = 1000
    BATCH_SIZE = 4
    UPDATE_RULE = 10
    for e in range(N_EPISODES):
        env.reset()

        while not env.done or sum(env.board) == 42:  # second condition never True!
            previous_state = deepcopy(env.state)
            action = agent.act(env.state)
            reward = env.update_state(action)

            replay.store(previous_state, action, reward, env.state)

            if len(replay.memory) >= BATCH_SIZE:
                transitions = replay.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                # boolean map on whether the state is terminal
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)))
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

                # Q(s_t, a_t)
                state_action_values = policy_net(state_batch).gather(1, action_batch)

                next_state_values = torch.zeros(BATCH_SIZE)
                next_state_values[non_final_mask] = \
                    target_net(non_final_next_states).max(1)[0].detach()
                # Q values of the target network for the same states (if they did not terminate)
                expected_state_action_values = (next_state_values * 0.999) + reward_batch

                loss = lossfn(state_action_values, expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()

                # gradient clipping
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

        if e % UPDATE_RULE == 0:
            target_net.load_state_dict(policy_net.state_dict())
