import torch

from ..common.replay import Dataset
from ..common.utils import state_action_append


class TransitionModelTrainer:
    def __init__(
        self,
        config: dict,
        transition_network: torch.nn.Module,
        autoencoder: torch.nn.Module,
        terminal_network: torch.nn.Module,
        batch_size: int,
        num_actions: int,
        device: str,
    ):
        self.device = device
        self.num_actions = num_actions

        self.window_length = config["window_length"]
        self.training_iterations = config["training_iterations"]

        self.autoencoder = autoencoder
        self.terminal_network = terminal_network
        self.transition_network = transition_network
        self.networks = [self.transition_network, self.terminal_network]

        self.prev_states = torch.zeros(batch_size, config["gru_dim"]).to(self.device)

    def train_(self, dataset: Dataset):
        """
        Update the recurrent transition model and the terminal model simultaneously using B sequences of length L for
        the specified number of training iterations. Gradients are accumulated for the specified window length, after
        which they are back-propagated.
        """
        for _ in range(self.training_iterations):
            o, a, o1, r, t = dataset.sample_sequences()
            length = len(o[0])

            self.prev_states = torch.zeros(
                dataset.batch_size, self.transition_network.gru_dim, device=self.device
            )
            for net in self.networks:
                net.loss = 0
            window_idx = 0
            for idx in range(length):
                for net in self.networks:
                    net.optimizer.zero_grad()

                s = self.autoencoder.embed(o[:, idx])
                s1 = self.autoencoder.embed(o1[:, idx])
                s_a = state_action_append(s, a[:, idx], self.num_actions, self.device)
                s1_r_pred, self.prev_states = self.transition_network.forward(
                    s_a, self.prev_states
                )
                t_pred = self.terminal_network.forward(s1)

                self.transition_network.loss += self.transition_network.loss_function(
                    torch.cat((s1, r[:, idx]), dim=1), s1_r_pred
                )
                self.terminal_network.loss += self.terminal_network.loss_function(
                    t[:, idx], t_pred
                )

                if window_idx == self.window_length or idx == length - 1:
                    for net in self.networks:
                        net.loss /= window_idx + 1
                        net.loss.backward()
                        net.optimizer.step()
                    self.prev_states = self.prev_states.detach()

                    dataset.logger.add_scalars(
                        ["Loss/Transition", "Loss/Terminal"],
                        [
                            self.transition_network.loss.item(),
                            self.terminal_network.loss.item(),
                        ],
                    )

                    self.transition_network.loss = 0
                    self.terminal_network.loss = 0
                    window_idx = 0
                else:
                    window_idx += 1
