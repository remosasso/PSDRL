from ..common.replay import Dataset
from ..networks.representation import AutoEncoder


class RepresentationTrainer:
    def __init__(self, training_iterations: int, ae: AutoEncoder):
        self.ae = ae
        self.training_iterations = training_iterations

    def train_(self, dataset: Dataset):
        for _ in range(self.training_iterations):
            o, _, _, _, _ = dataset.sample_sequences()
            self.ae.loss = 0
            for sequence in o:
                self.ae.optimizer.zero_grad()
                s = self.ae.encoder.forward(sequence)
                decode_s = self.ae.decoder.forward(s)
                self.ae.loss = self.ae.loss_function(sequence, decode_s)
                self.ae.loss.backward()
                self.ae.optimizer.step()

        dataset.logger.add_scalars("Loss/AE", self.ae.loss.item())
