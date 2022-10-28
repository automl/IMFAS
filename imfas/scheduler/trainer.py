from imfas.scheduler.agent import DQNAgent
from imfas.scheduler.environment import Environment


class SchedulerTrainer:
    def __init__(
            self,
            optimizer,
            agent: DQNAgent, environment:
            Environment,
            loss_fn
    ):
        self.agent = agent
        self.optimizer = optimizer(agent.parameters())
        self.environment = environment
        self.loss_fn = loss_fn

    def train(self, n_epochs, batch_size, ):
        self.optimize_policy(batch_size)

    def optimize_policy(self, batch_size):
        # sample batch from replay memory
        # compute loss
        # optimize policy

        # Computeloss
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.agent.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        pass
