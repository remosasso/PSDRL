from copy import deepcopy

import torch

from ..bayes.neural_linear_model import NeuralLinearModel
from ..common.replay import Dataset
from ..common.settings import TP_THRESHOLD


class PolicyTrainer:
    def __init__(
        self,
        config: dict,
        gru_dim: int,
        value_network: torch.nn.Module,
        device: str,
        batch_size: int,
        actions: torch.tensor,
    ):
        self.target_net = None

        self.device = device
        self.actions = actions
        self.gru_dim = gru_dim
        self.batch_size = batch_size
        self.value_network = value_network

        self.discount = config["discount"]
        self.target_update_freq = config["target_update_freq"]
        self.training_iterations = config["training_iterations"]

        self.num_actions = len(actions)
        self.prev_states = torch.zeros(batch_size, self.gru_dim, device=self.device)

    def train_iter(self, inputs: torch.tensor, targets: torch.tensor):
        self.value_network.optimizer.zero_grad()

        outputs = self.value_network.forward(inputs)
        self.value_network.loss = self.value_network.loss_function(outputs, targets)

        self.value_network.loss.backward()
        self.value_network.optimizer.step()

    def simulate(
        self,
        states: torch.tensor,
        actions: torch.tensor,
        h: torch.tensor,
        model: NeuralLinearModel,
    ):
        """
        Simulate one timestep using the sampled model, and retain the hidden states (h) that correspond to the actions
        taken as dictated by the sampled sequences for the next timestep.
        """
        s1, r, t, h1 = model.predict(states, h)

        h = h1[
            (torch.arange(0, len(states)) * self.num_actions).to(self.device)
            + actions.reshape(-1).long()
        ]
        return (
            r.reshape(-1, self.num_actions),
            s1,
            t.reshape(-1, self.num_actions),
            h,
            h1,
        )

    def compute_targets(
        self,
        rewards: torch.tensor,
        next_states: torch.tensor,
        terminals: torch.tensor,
        target_net: torch.nn.Sequential,
    ):
        v = target_net(next_states).detach().flatten().reshape(-1, self.num_actions)
        y, _ = torch.max(
            rewards + (self.discount * v * (terminals < TP_THRESHOLD)), dim=1
        )
        return y.reshape(-1, 1)

    def train_(self, model: NeuralLinearModel, dataset: Dataset):
        """
        Update the value network using B sequences of length L for the specified number of training iterations (kappa).
        For the states in all B sequences at a given timestep, the next states and rewards are simulated in parallel
        using the current sampled model. The simulated elements are used to compute the targets with which the value
        network is updated.
        """
        for epoch in range(self.training_iterations):
            o, a, _, _, _ = dataset.sample_sequences()
            length = len(o[0])

            self.prev_states[:] = torch.zeros(self.batch_size, self.gru_dim)
            for idx in range(length):
                if idx % self.target_update_freq == 0:
                    self.target_net = deepcopy(self.value_network.layers)

                s = model.autoencoder.embed(o[:, idx])
                inputs = torch.cat((s, self.prev_states), dim=1)

                rewards, next_states, terminals, self.prev_states, h1 = self.simulate(
                    s, a[:, idx], self.prev_states, model
                )
                targets = self.compute_targets(
                    rewards,
                    torch.cat((next_states, h1), dim=1),
                    terminals,
                    self.target_net,
                )

                self.train_iter(inputs, targets)

        dataset.logger.add_scalars("Loss/Value", self.value_network.loss.item())
