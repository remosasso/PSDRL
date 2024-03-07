import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.logger import Logger
from ..common.utils import extract_episode_data, preprocess
from ..common.settings import STATE_SIZE


class Dataset:
    def __init__(
        self,
        logger: "Logger",
        config: dict,
        time_limit: int,
        device: str,
        seed: int = None,
    ):

        self.cum_rew = 0
        self.add_idx = 0
        self.n_samples = 0
        self.max_ep_len = 0
        self.min_ep_len = 0
        self.episode_add_idx = 0
        self.total_num_transitions = 0
        self.replace = False
        self.episodes = []
        self.logger = logger
        self.device = device
        self.random_state = np.random.RandomState(seed)

        self.capacity = config["capacity"]
        self.batch_size = config["batch_size"]
        self.sequence_len = config["sequence_length"]

        self.tmp_episode = {
            "states": torch.zeros(
                (time_limit + 1, STATE_SIZE, STATE_SIZE),
                dtype=torch.uint8,
                device=self.device,
            ),
            "actions": torch.zeros((time_limit + 1, 1), device=self.device),
            "next_states": torch.zeros(
                (time_limit + 1, STATE_SIZE, STATE_SIZE),
                dtype=torch.uint8,
                device=self.device,
            ),
            "rewards": torch.zeros((time_limit + 1, 1), device=self.device),
            "terminals": torch.zeros((time_limit + 1, 1), device=self.device),
        }

    def add_data(
        self,
        new_state: np.array,
        new_action: int,
        new_next_state: np.array,
        new_reward: int,
        new_terminal: bool,
    ):
        """
        Add new transition to a temporary episode until a terminal state is reached. Then, append episode to replay
        buffer or replace it with the episode at self.episode_add_idx if capacity is reached.
        """
        self.tmp_episode["states"][self.add_idx] = torch.from_numpy(
            preprocess(new_state)
        ).type(torch.uint8)
        self.tmp_episode["actions"][self.add_idx] = new_action
        self.tmp_episode["next_states"][self.add_idx] = torch.from_numpy(
            preprocess(new_next_state)
        ).type(torch.uint8)
        self.tmp_episode["rewards"][self.add_idx] = torch.tensor(np.tanh(new_reward))
        self.tmp_episode["terminals"][self.add_idx] = torch.tensor(new_terminal)

        self.add_idx += 1
        self.n_samples += 1
        self.cum_rew += new_reward

        if new_terminal:
            new_episode = {
                "states": self.tmp_episode["states"][: self.add_idx].clone(),
                "actions": self.tmp_episode["actions"][: self.add_idx].clone(),
                "next_states": self.tmp_episode["next_states"][: self.add_idx].clone(),
                "rewards": self.tmp_episode["rewards"][: self.add_idx].clone(),
                "terminals": self.tmp_episode["terminals"][: self.add_idx].clone(),
                "cum_rew": self.cum_rew,
            }
            if self.n_samples > self.capacity:
                self.n_samples = 0
                self.episode_add_idx = 0
                self.replace = True

            if self.replace and self.episode_add_idx < len(self.episodes):
                self.episodes[self.episode_add_idx] = new_episode
            else:
                self.episodes.append(new_episode)
            self.episode_add_idx += 1

            (
                self.min_ep_len,
                self.max_ep_len,
                self.total_num_transitions,
            ) = self.logger.add_replay_statistics(self.episodes)

            self.add_idx = 0
            self.cum_rew = 0

    def sample_sequences(self):
        """
        Sample B sequences of length L, where L is set to the current minimum episode length if no episodes exist that
        last longer than L. A sequence is sampled by sampling a random episode, then sampling a random valid
        timestep within that episode, and finally extracting the sequence episode[timestep, ..., timestep+L].
        """
        batches = []
        sequence_length = (
            self.sequence_len
            if self.max_ep_len > self.sequence_len
            else self.min_ep_len
        )
        for _ in range(self.batch_size):
            ep_len = 0
            while ep_len < sequence_length:
                ep = self.random_state.randint(0, len(self.episodes))
                ep_len = len(self.episodes[ep]["states"])
            timestep = self.random_state.randint(0, ep_len - sequence_length + 1)
            batches.append(
                {
                    "states": self.episodes[ep]["states"][
                        timestep : timestep + sequence_length
                    ],
                    "actions": self.episodes[ep]["actions"][
                        timestep : timestep + sequence_length
                    ],
                    "next_states": self.episodes[ep]["next_states"][
                        timestep : timestep + sequence_length
                    ],
                    "rewards": self.episodes[ep]["rewards"][
                        timestep : timestep + sequence_length
                    ],
                    "terminals": self.episodes[ep]["terminals"][
                        timestep : timestep + sequence_length
                    ],
                }
            )
        return extract_episode_data(batches)
