import torch

# General
STATE_SIZE = 64
BATCH_EMBEDDING_SIZE = 1000

# Bayesian linear regression
BLR_COEFFICIENT = 100
ONE_OVER_LAMBDA = 1

# Autoencoder
CONV_FEAT = 48
KERNEL = 4
STRIDE = 2
AE_LOSS_F = torch.nn.MSELoss()
AE_OPTIM = torch.optim.Adam

# Terminal predictor
TP_THRESHOLD = 0.5
TP_LOSS_F = torch.nn.MSELoss()
TP_OPTIM = torch.optim.Adam

# Transition model
REC_CELL = torch.nn.GRUCell
TM_LOSS_F = torch.nn.MSELoss()
TM_OPTIM = torch.optim.Adam

# Value net
VN_LOSS_F = torch.nn.MSELoss()
VN_OPTIM = torch.optim.Adam

# Atari
NOOP = 30
FRAME_SKIP = 4
