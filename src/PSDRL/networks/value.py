import torch
import torch.nn as nn

from ..common.settings import VN_LOSS_F, VN_OPTIM


class Network(nn.Module):
    def __init__(self, input_dim: int, config: dict, device: str, gru_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim + gru_dim, config["hidden_dim"] + gru_dim),
            nn.Tanh(),
            nn.Linear(config["hidden_dim"] + gru_dim, config["hidden_dim"] + gru_dim),
            nn.Tanh(),
            nn.Linear(config["hidden_dim"] + gru_dim, config["hidden_dim"] + gru_dim),
            nn.Tanh(),
            nn.Linear(config["hidden_dim"] + gru_dim, config["hidden_dim"] + gru_dim),
            nn.Tanh(),
            nn.Linear(config["hidden_dim"] + gru_dim, 1),
        )
        self.loss_function = VN_LOSS_F
        self.optimizer = VN_OPTIM(self.parameters(), lr=config["learning_rate"])
        self.to(device)
        self.loss = 0

    def forward(self, x: torch.tensor):
        return self.layers(x)

    def predict(self, x: torch.tensor):
        with torch.no_grad():
            return self.layers(x)
