import torch
import torch.nn as nn

from ..common.settings import TP_LOSS_F, TP_OPTIM


class Network(nn.Module):
    def __init__(self, input_dim: int, config: dict, device: str):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, config["hidden_dim"]),
            nn.Tanh(),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.Tanh(),
            nn.Linear(config["hidden_dim"], config["hidden_dim"]),
            nn.Tanh(),
            nn.Linear(config["hidden_dim"], 1),
        )
        self.optimizer = TP_OPTIM(self.parameters(), lr=config["learning_rate"])
        self.loss_function = TP_LOSS_F
        self.loss = 0
        self.to(device)

    def forward(self, x: torch.tensor):
        return self.layers(x)

    def predict(self, x: torch.tensor):
        with torch.no_grad():
            return self.layers(x)
