# Posterior Sampling for Deep Reinforcement Learning

Implementation of Posterior Sampling for Deep Reinforcement Learning [(PSDRL)](https://arxiv.org/abs/2305.00477) in PyTorch.

```
@inproceedings{sasso2023posterior,
  title = {Posterior Sampling for Deep Reinforcement Learning},
  author = {Sasso, Remo and Conserva, Michelangelo and Rauber, Paulo},
  booktitle={International Conference on Machine Learning},
  year = {2023}
}
```

## Overview

PSDRL is the first truly scalable approximation of Posterior Sampling for Reinforcement Learning (PSRL) that retains its model-based essence. In the Atari benchmark, PSDRL significantly outperforms previous state-of-the-art attempts at scaling up posterior sampling such as Bootstrapped DQN + Priors and Successor Uncertainties, while being strongly competitive with the state-of-the-art DreamerV2 agent, both in sample efficiency and computational efficiency. 

PSDRL maps high-dimensional observations to a low-dimensional continuous latent state using an autoencoder (a) that enables predicting transitions in latent state space for any given action using a recurrent transition model (b).

![Continuous Latent Space Transition Model](https://imgur.com/XfLun7N.png)

PSDRL represents uncertainty through a Bayesian neural network that maintains a distribution over the parameters of the last layer of the transition model, which allows PSDRL to sample a model of the environment.
Planning w.r.t. the sampled model is carried out with a value network that is fitted using predictions from the sampled model, thereby approximating the optimal policy w.r.t. the sampled model.
The agent then collects data by acting greedily w.r.t. the current sampled model and value network.

<p align="center">
<img src="https://imgur.com/AoCb3g9.png" width=50% height=50%>
</p>

By acting greedily w.r.t. different sampled models, the exploration of the agent is naturally driven through uncertainty over models of the environments.
An example of trajectories predicted with different sampled models can be found below.
Although each trajectory starts from the same initial state and uses identical parameters for the neural network components, it is possible to notice a remarkable diversity among the different sampled models.

<p align="center">
<img src="https://imgur.com/EcnPvtr.gif">
</p>

For further details, results, and comparisons see the [research paper](https://arxiv.org/pdf/2305.00477.pdf).

## Instructions

Install the dependencies:
```
pip install -r requirements.txt
```

You can run the PSDRL agent by calling the [main.py](https://github.com/remosasso/PSDRL-new/blob/master/src/main.py) file, which accepts a configuration file (in the yaml format) and the code name corresponding to the Atari game.
For example, you can run the PSDRL agent on _Pong_ with the parameters from the paper as,
```
python src/main.py --config="src/config.yaml" --env="Pong"
```
You can set a fixed seed with an additional parameter, e.g. `--seed 42`.

Training can be monitored with Tensorboard.
```
tensorboard --logdir=src/logdir
```

## Environments

The repository includes the Atari games.
If you wish to test the algorithm on different environments, you can add them to the `init_env` function in the [utils.py](https://github.com/remosasso/PSDRL-new/blob/master/src/PSDRL/common/utils.py#L18) file.

The implementation targets environments with visual observation that are grayscale in range zero-one and of dimension 64x64, so please make sure that the observations of environments correspond to this format.

If you wish to test the algorithm on environments with vectorial observations, you can either implement a different architecture for the autoencoder (see [representation.py](https://github.com/remosasso/PSDRL-new/blob/master/src/PSDRL/networks/representation.py)) or remove the autoencoder altogether.

Feel free to reach out if you need any help.


## Runtime
The implementation found in this repository runs on a single GPU and takes about 8 and 15 hours per 1M environment steps in Atari on an NVIDIA A100 and NVIDIA V100 GPU, respectively. The table below shows the expected runtime for an A100 GPU.

| Game     | Runtime         |
|----------|---------------|
| Freeway  | 8h53m $\pm$ 0m |
| Qbert    | 7h39m $\pm$ 43m |
| Enduro   | 9h35m $\pm$ 15m |
| Asterix  | 7h44m $\pm$ 24m |
| Seaquest | 8h31m $\pm$ 2m  |
| Pong     | 7h58m $\pm$ 42m |
| Hero     | 9h26m $\pm$ 2m  |
| **Average**  | **8h31m $\pm$ 44m** |

 See Appendix E of the paper for a comparison with the baselines. 


## Tips

You can track additional metrics by calling the `add_scalars` function of the [Logger](https://github.com/remosasso/PSDRL-new/blob/master/src/PSDRL/common/logger.py) object.
