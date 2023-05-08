import torch
from torch import nn

from ..common.settings import REC_CELL, TM_LOSS_F, TM_OPTIM


class Network(nn.Module):
    def __init__(self, embed_dim: int, n_actions: int, config: dict, device: str):
        super().__init__()

        self.gru_dim = config["gru_dim"]
        self.latent_dim = self.gru_dim + config["hidden_dim"]

        self.layers = nn.Sequential(
            nn.Linear(self.gru_dim + embed_dim + n_actions, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, embed_dim + 1),
        )
        self._cell = REC_CELL(embed_dim + n_actions, self.gru_dim)
        self.loss_function = TM_LOSS_F
        self.optimizer = TM_OPTIM(self.parameters(), lr=config["learning_rate"])
        self.to(device)
        self.loss = 0

    def forward(self, x: torch.tensor, hidden: torch.tensor):
        h = self._cell(x, hidden)
        return self.layers(torch.cat((h, x), dim=1)), h

    def predict(self, x: torch.tensor, hidden: torch.tensor):
        with torch.no_grad():
            return self.forward(x, hidden)
