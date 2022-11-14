import functools
import random
from collections import deque, namedtuple
from copy import deepcopy

import math
import torch
import wandb
from torch import nn, optim
from torch.nn import HuberLoss


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
            return torch.tensor([[random.randrange(7)]],  # n_actions # todo: read from policy_net
                                device=self.device,
                                dtype=torch.long)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class Replaybuffer:
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def __init__(self, capacity, device: str = "cpu"):
        self.memory = deque(maxlen=capacity)
        self.device = device

    def store(self, *args):
        self.memory.append(self.Transition(*args))
        # Consider, that we could collect those transitions, that are being removed
        #  in favour of new ones and store them to a separate buffer, that once
        #  filled will write to disk. This will create data batches, from which
        #  we can train our (or another model). Alleviates to always interact with the
        #  environment. Can be used to pretrain the policy network of another model.

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)

        # parse the batch into separate tensors
        batch = self.Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # boolean map on whether the state is terminal
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state))
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # move to device
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        non_final_next_states = non_final_next_states.to(self.device)
        non_final_mask = non_final_mask.to(self.device)

        return state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask

    def __len__(self):
        return len(self.memory)


class EnvironmentCumulativeAction:
    def __init__(self):
        # todo accept performance & cost tensors for multiple datasets.
        self._done = False
        self.cumulative_reward = 0

        # state: 4 wins board
        self.board = torch.zeros(7, 6)

    def __repr__(self):
        return str(self.board)

    @property
    def done(self):
        """Check if the game is over."""
        if torch.any(self.board.sum(axis=1) == 4) or sum(self.board) == 42:
            self._done = True
        else:
            self._done = False

        return self._done

    @property
    def state(self):
        if self.done:
            return None  # for termination mask in replay!
        # TODO mask, performance tensor, cummulative cost # optionally:
        #  imfas(dataset_metaf, learning curves, mask)
        return torch.cat([self.board.view(1, -1), self.board.sum()])  # append current cost.

    def update_state(self, action: int):
        column_count = self.board.sum(axis=1)
        self.board[6 - column_count[action], action] = 1

    def reward(self):
        # TODO: -w * regret + (1-w) * cumulative cost
        reward = -1  # for every step

        if self.done:
            reward = 1

        self.cumulative_reward += reward

        wandb.log({"cumulative_reward": self.cumulative_reward})
        return reward

    def reset(self):
        # TODO change dataset, that is acted on.
        self.board = torch.zeros(7, 6)
        self._done = False
        self.cumulative_reward = 0


class EnvironmentAdditiveAction(EnvironmentCumulativeAction):
    """
    Fidelity: training subset size; the agent can choose from any of the coordinates
    on the board
    """
    pass


class EnvironmentRaw(EnvironmentCumulativeAction):
    """
    State is the masking ("board") and the raw performances. The agent will have to
    formulate a ranking as part of the action, so that we can formulate a reward based on the
    regret. Episode ends if either budget is exceeded or the agent has found the best.
    """
    pass


class DQNTrainer:
    def __init__(self, agent, environment, replay, optimizer: functools.partial, loss_fn):
        self.agent = agent
        self.env = environment
        self.replay = replay
        self.optimizer = optimizer(self.agent.policy_net.parameters())
        self.loss_fn = loss_fn

    def run(self, n_episodes, batch_size, gamma=0.999, target_update=10):
        """
        n_episodes: number of episodes to train for
        batch_size: number of transitions to sample from replay buffer
        gamma: discount factor
        target_update: number of steps before updating the target network
        """

        for e in range(n_episodes):
            self.env.reset()

            while not self.env.done:
                previous_state = deepcopy(self.env.state)
                action = self.agent.act(self.env.state)
                reward = self.env.update_state(action)

                self.replay.store(previous_state, action, reward, env.state)

                if len(replay.memory) >= batch_size:
                    # parse the replay buffer to create a batch
                    state_batch, action_batch, reward_batch, non_final_next_states, \
                    non_final_mask = self.replay.sample(batch_size)

                    # Q(s_t, a_t)
                    state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

                    next_state_values = torch.zeros(batch_size)
                    next_state_values[non_final_mask] = \
                        self.agent.target_net(non_final_next_states).max(1)[0].detach()
                    # Q values of the target network for the same states (if they did not terminate)
                    expected_state_action_values = \
                        (next_state_values * gamma) + reward_batch  # GAMMA?

                    loss = self.loss_fn(
                        state_action_values,
                        expected_state_action_values.unsqueeze(1)
                    )

                    wandb.log({"loss": loss})

                    self.optimizer.zero_grad()
                    loss.backward()

                    # gradient clipping
                    for param in self.agent.policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)

                    self.optimizer.step()

            if e % target_update == 0:
                self.agent.update_target_net()


if __name__ == '__main__':
    # TODO HPO: batch_size, episodes, update rule, lr, gamma, eps_decay, eps_start, eps_end,
    #  hidden_dims, activation

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

    trainer = DQNTrainer(
        agent=DQNAgent(policy_net, target_net, eps_start=0.9, eps_end=0.05, eps_decay=200),
        environment=EnvironmentCumulativeAction(),
        replay=Replaybuffer(1000),
        optimizer=functools.partial(optim.Adam, lr=1e-3),
        loss_fn=HuberLoss()
    )

    trainer.run(1000, 128)

    # TODO Test procedure

    # TODO Baselines / Benchmarks: Meta-Reveal. Same holdouts as for imfas?
