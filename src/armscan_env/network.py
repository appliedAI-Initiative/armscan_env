from collections.abc import Callable, Sequence
from typing import Any, Generic

import numpy as np
import torch
from tianshou.data.batch import BatchProtocol
from tianshou.utils.net.common import TRecurrentState
from torch import nn


class LabelmapsObsBatchProtocol(BatchProtocol):
    channelled_labelmap_BCWH: np.ndarray
    action_BA: np.ndarray
    reward_B: np.ndarray


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize a layer with the given standard deviation and bias constant."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN_MLP_Concat(nn.Module, Generic[TRecurrentState]):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        a: int,
        r: int,
        action_shape: Sequence[int] | int,
        device: str | int | torch.device = "cpu",
        features_only: bool = False,
        output_dim_added_layer: int | None = None,
        mlp_output_dim: int = 512,
        layer_init: Callable[[nn.Module], nn.Module] = layer_init,
    ) -> None:
        # if not features_only and output_dim_added_layer is not None:
        #     raise ValueError(
        #         "Should not provide explicit output dimension using `output_dim_added_layer` when `features_only` is true.",
        #     )
        super().__init__()
        self.device = device

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(a + r, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, mlp_output_dim),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            base_cnn_output_dim = int(np.prod(self.cnn(torch.zeros(1, c, h, w)).shape[1:]))
        base_mlp_output_dim = mlp_output_dim
        action_dim = int(np.prod(action_shape))
        concat_dim = base_cnn_output_dim + base_mlp_output_dim

        self.combined = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim),
        )
        self.output_dim = action_dim

        # if not features_only:
        #     action_dim = int(np.prod(action_shape))
        #     self.combined = nn.Sequential(
        #         self.combined,
        #         layer_init(nn.Linear(512, action_dim)),
        #     )
        #     self.output_dim = action_dim

        # elif output_dim_added_layer is not None:
        #     self.combined = nn.Sequential(
        #         self.combined,
        #         layer_init(nn.Linear(512, output_dim_added_layer)),
        #         nn.ReLU(inplace=True),
        #     )
        #     self.output_dim = output_dim_added_layer
        # else:
        #     self.output_dim = base_cnn_output_dim

    def forward(
        self,
        obs: LabelmapsObsBatchProtocol,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        channeled_labelmap = torch.as_tensor(
            obs.channelled_labelmap_BCWH,
            device=self.device,
            dtype=torch.float32,
        )
        image_output = self.cnn(channeled_labelmap)

        action_reward = torch.concat(
            [torch.from_numpy(obs.action_BA), torch.from_numpy(obs.reward_B)],
            dim=0,
        )
        action_reward_output = self.mlp(action_reward)

        concat = torch.cat([image_output, action_reward_output], dim=0)
        return self.combined(concat), state
