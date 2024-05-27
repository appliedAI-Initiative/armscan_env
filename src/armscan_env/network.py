from collections.abc import Callable
from typing import Any, Generic

import numpy as np
import torch
from torch import nn

from tianshou.data.batch import BatchProtocol
from tianshou.utils.net.common import TRecurrentState


# KEEP IN SYNC WITH ChanneledLabelmapsObsWithActReward
# This batch will be created automatically from these observations
class LabelmapsObsBatchProtocol(BatchProtocol):
    """Batch protocol for the observation of the LabelmapSliceAsChannelsObservation class.
    Must have the same fields as the TDict of ChanneledLabelmapsObsWithActReward.
    """

    channeled_slice: np.ndarray
    action: np.ndarray
    reward: np.ndarray


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize a layer with the given standard deviation and bias constant."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN_MLP_Concat(nn.Module, Generic[TRecurrentState]):
    """A composed network for DQN with a CNN for the channeled slice observation and an MLP for the action-reward
    observation.
    The CNN is composed of 3 convolutional layers with ReLU activation functions.

    * input: channeled slice observation,
    * first layer: 32 filters with kernel size 8 and stride 4,
    * second layer: 64 filters with kernel size 4 and stride 2,
    * third layer: 64 filters with kernel size 3 and stride 1.
    * output: flattened output.

    The MLP is composed of 2 linear layers with ReLU activation functions.

    * input: last action and previous reward concatenated,
    * hidden layer: 512 units,
    * output layer: mlp_output_dim units.

    The final processing MLP is composed of 3 linear layers with ReLU activation functions.

    * input: concatenation of the CNN and MLP outputs,
    * first layer: 512 units,
    * second layer: 512 units,
    * output layer: action_dim units.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_dim: int,
        device: str | int | torch.device = "cpu",
        mlp_output_dim: int = 512,
        layer_init: Callable[[nn.Module], nn.Module] = layer_init,
    ) -> None:
        super().__init__()
        self.device = device

        self.channeled_slice_cnn_CHW = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        mlp_input_dim = action_dim + 1  # action concatenated with reward
        self.action_reward_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, mlp_output_dim),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            base_cnn_output_dim = int(
                np.prod(self.channeled_slice_cnn_CHW(torch.zeros(1, c, h, w)).shape[1:]),
            )
        base_mlp_output_dim = mlp_output_dim
        concat_dim = base_cnn_output_dim + base_mlp_output_dim

        self.final_processing_mlp = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim),
        )
        self.output_dim = action_dim

    def forward(
        self,
        obs: LabelmapsObsBatchProtocol,
        state: Any | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*).

        This method is used to generate the Q value from the given input data.
        * The channeled_slice observation is passed through a CNN,
        * The last action and previous reward are concatenated and passed through an MLP,
        * The outputs of the CNN and MLP are concatenated and passed through a final MLP.
        The output of the final MLP is the Q value of each action.
        """
        channeled_slice = torch.as_tensor(obs.channeled_slice, dtype=torch.float, device=self.device)
        image_output = self.channeled_slice_cnn_CHW(channeled_slice)

        action_reward = torch.concat(
            [torch.as_tensor(obs.action, device=self.device), torch.as_tensor(obs.reward, device=self.device)],
            dim=1,
        )
        action_reward_output = self.action_reward_mlp(action_reward)

        concat = torch.cat([image_output, action_reward_output], dim=1)
        return self.final_processing_mlp(concat), state
