import os
import pickle
from typing import TYPE_CHECKING

import gym
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color

from .settings import STATE_SIZE, FRAME_SKIP, NOOP

if TYPE_CHECKING:
    from ..agent.psdrl import PSDRL


def init_env(suite: str, env: str, test: bool):
    if suite == "atari":
        full_game_name = "{}NoFrameskip-v4".format(env)
        environment = gym.make(full_game_name)
        environment = gym.wrappers.AtariPreprocessing(
            environment, NOOP, FRAME_SKIP, STATE_SIZE, False, True, scale_obs=True
        )

        test_environment = None
        if test:
            test_environment = gym.make(full_game_name)
            test_environment = gym.wrappers.AtariPreprocessing(
                test_environment,
                NOOP,
                FRAME_SKIP,
                STATE_SIZE,
                False,
                True,
                scale_obs=True,
            )
    else:
        raise NotImplementedError(f"{suite} is not available.")

    return environment, list(range(environment.action_space.n)), test_environment


def preprocess_image(image: np.array):
    """
    This function preprocesses images such that they become 64x64 grayscale images to ensure compatibility with
    PSDRL.
    """
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    elif len(image.shape) < 2:
        raise NotImplementedError("Observation is not an image!")
    if image.shape[0] != STATE_SIZE or image.shape[1] != STATE_SIZE:
        image = Image.fromarray(image)
        image = image.resize((STATE_SIZE, STATE_SIZE), Image.NEAREST)
    image = np.array(image)
    if np.max(image) > 1:
        image /= 255
    return image


def expand_image(img: torch.tensor):
    extra_dim = [1] if len(img.shape) == 3 else [1, 1]
    return img.expand(tuple(extra_dim) + tuple(img.shape))


def state_action_append(
    obs: torch.tensor, action: torch.tensor, n_actions: int, device: str
):
    return torch.hstack(
        (obs, F.one_hot(action.reshape(-1).long(), num_classes=n_actions).to(device))
    )


def create_state_action_batch(
    states: torch.tensor,
    actions: torch.tensor,
    h: torch.tensor,
    n_actions: int,
    device: str,
):
    """
    Create a batch of states and hidden states for all possible actions.
    """
    h_dim = h.shape[0] if len(h.shape) == 1 else h.shape[1]
    state_dim = states[0].shape[0]
    states_x = torch.repeat_interleave(states.T, n_actions).reshape(state_dim, -1).T
    actions_x = torch.tile(actions, (1, len(states))).reshape(-1, 1)
    state_action_x = state_action_append(states_x, actions_x, n_actions, device)
    h_x = torch.repeat_interleave(h.T, n_actions).reshape(h_dim, -1).T
    return state_action_x, h_x


def create_directories(env: str, algorithm: str, name: str):
    if not os.path.exists("./logdir"):
        os.mkdir("./logdir/")
    env_folder = "./logdir/{}".format(env)
    if not os.path.exists(env_folder):
        os.mkdir(env_folder)
    folder_name = env_folder + "/" + algorithm + "-{}".format(name)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    number = len(os.listdir(folder_name))
    # Avoid duplicate
    while os.path.exists(folder_name + "/{}/".format(number)):
        number += 1
    logdir = folder_name + "/{}/".format(number)
    os.mkdir(logdir)
    os.mkdir(logdir + "checkpoints/")
    return logdir


def load(agent: "PSDRL", load_dir: str):
    agent.model.transition_network.load_state_dict(
        torch.load(load_dir + "transition.pt")
    )
    agent.model.terminal_network.load_state_dict(torch.load(load_dir + "terminal.pt"))
    agent.model.autoencoder.load_state_dict(torch.load(load_dir + "autoencoder.pt"))
    agent.value_network.load_state_dict(torch.load(load_dir + "value.pt"))
    with open(load_dir + "replay.pt".format(load_dir), "rb") as fn:
        agent.dataset.episodes = pickle.load(fn)
    print("Successfully loaded.")


def preprocess(obs):
    return obs * 255 if len(obs.shape) > 1 else obs


def extract_episode_data(episodes: list):
    o = torch.stack([ep["states"] for ep in episodes])
    a = torch.stack([ep["actions"] for ep in episodes])
    o1 = torch.stack([ep["next_states"] for ep in episodes])
    r = torch.stack([ep["rewards"] for ep in episodes])
    t = torch.stack([ep["terminals"] for ep in episodes])
    return (
        o / (255 if len(o.shape) > 3 else 1),
        a,
        o1 / (255 if len(o1.shape) > 3 else 1),
        r,
        t,
    )
