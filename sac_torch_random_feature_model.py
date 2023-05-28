import logging
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional

from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.models.catalog import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.sac import SACConfig
from utils import angle_normalize

# torch, nn = try_import_torch()
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 256,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'quadrotor_2d',
                                      'dynamics_parameters': {'stabilizing_target': torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
                                                              },
                                      'sin_input': True}

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)


class RandomFeatureNetwork(TorchModelV2, nn.Module):
    """
    Random feature network.
    """

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        prev_layer_size = int(np.product(obs_space.shape))
        random_feature_dim = model_config.get('random_feature_dim')
        self.random_feature_dim = random_feature_dim

        fourier_random_feature = nn.Linear(prev_layer_size, random_feature_dim)
        if model_config.get('sigma') > 0:
            nn.init.normal_(fourier_random_feature.weight, std=1. / model_config.get('sigma'))
        else:
            nn.init.normal_(fourier_random_feature.weight)
        nn.init.uniform_(fourier_random_feature.bias, 0, 2. * np.pi)
        learn_rf = model_config.get('learn_rf', False)
        fourier_random_feature.weight.requires_grad = learn_rf
        fourier_random_feature.bias.requires_grad = learn_rf
        self.fournier_random_feature = fourier_random_feature

        linear1 = nn.Linear(random_feature_dim, 1)
        linear1.bias.requires_grad = False
        self.linear1 = linear1
        self.dynamics_type = model_config.get('dynamics_type')
        if self.dynamics_type == 'quadrotor_2d':
            self.stabilizing_target = model_config.get('stabilizing_target')
        self.dynamics_parameters = model_config.get('dynamics_parameters')
        self.sin_input = model_config.get('sin_input', False)
        # self.device = torch.device('cuda' if torch.cuda.is_avaliable() else 'cpu')

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        x = torch.cos(self.fournier_random_feature(self._last_flat_in))
        x = torch.div(x, 1. / self.random_feature_dim)
        logits = self.linear1(x)
        return logits, state

    def value_function(self) -> TensorType:
        raise NotImplementedError


class SACTorchRFModel(SACTorchModel):

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: Optional[int],
            model_config: ModelConfigDict,
            name: str,
            policy_model_config: ModelConfigDict = None,
            q_model_config: ModelConfigDict = None,
            twin_q: bool = False,
            initial_alpha: float = 1.0,
            target_entropy: Optional[float] = None,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, policy_model_config, q_model_config,
                         twin_q, initial_alpha, target_entropy)

        if isinstance(action_space, Discrete):
            action_outs = q_outs = self.action_dim
        elif isinstance(action_space, Box):
            q_outs = 1
        else:
            assert isinstance(action_space, Simplex)
            q_outs = 1

        self.q_net = RandomFeatureNetwork(obs_space, action_space, q_outs, q_model_config, 'q1')

        if twin_q:
            self.twin_q_net = RandomFeatureNetwork(obs_space, action_space, q_outs, q_model_config, 'q1')

    def _get_q_value(self, model_out, actions, net):
        # Model outs may come as original Tuple observations, concat them
        # here if this is the case.
        if isinstance(net.obs_space, Box):
            if isinstance(model_out, (list, tuple)):
                model_out = torch.cat(model_out, dim=-1)
            elif isinstance(model_out, dict):
                model_out = torch.cat(list(model_out.values()), dim=-1)

        # Continuous case -> concat actions to model_out.
        if actions is not None:
            if self.q_net.dynamics_type == 'quadrotor_2d':
                if self.q_net.sin_input:
                    obs_tp1 = self.quadrotor_f_star_7d(model_out, actions)
                else:
                    obs_tp1 = self.quadrotor_f_star_6d(model_out, actions)
            elif self.q_net.dynamics_type == 'pendulum':
                obs_tp1 = self.pendulum_3d(model_out, actions)
            input_dict = {"obs": obs_tp1}
        # Discrete case -> return q-vals for all actions.
        else:
            input_dict = {"obs": model_out}
        # Switch on training mode (when getting Q-values, we are usually in
        # training).
        input_dict["is_training"] = True

        v_tp1, state_out = net(input_dict, [], None)
        q_t = self._get_reward(model_out, actions) + v_tp1

        return q_t, state_out

    def pendulum_3d(self, obs, action, g=10.0, m=1., l=1., max_a=2., max_speed=8., dt=0.05):
        th = torch.atan2(obs[:, 1], obs[:, 0])  # 1 is sin, 0 is cosine
        thdot = obs[:, 2]
        action = torch.reshape(action, (action.shape[0],))
        u = torch.clip(action, -max_a, max_a)
        newthdot = thdot + (3. * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = torch.clip(newthdot, -max_speed, max_speed)
        newth = th + newthdot * dt
        new_states = torch.empty((obs.shape[0], 3))
        new_states[:, 0] = torch.cos(newth)
        new_states[:, 1] = torch.sin(newth)
        new_states[:, 2] = newthdot
        return new_states

    def quadrotor_f_star_6d(self, states, action, m=0.027, g=10.0, Iyy=1.4e-5, dt=0.0167):
        dot_states = torch.empty_like(states)
        dot_states[:, 0] = states[:, 1]
        dot_states[:, 1] = 1 / m * torch.multiply(torch.sum(action, dim=1), torch.sin(states[:, 4]))
        dot_states[:, 2] = states[:, 3]
        dot_states[:, 3] = 1 / m * torch.multiply(torch.sum(action, dim=1), torch.cos(states[:, 4])) - g
        dot_states[:, 4] = states[:, 5]
        dot_states[:, 5] = 1 / 2 / Iyy * (action[:, 1] - action[:, 0])

        new_states = states + dt * dot_states

        return new_states

    def quadrotor_f_star_7d(self, states, action, m=0.027, g=10.0, Iyy=1.4e-5, dt=0.0167):
        new_states = torch.empty_like(states)
        new_states[:, 0] = states[:, 0] + dt * states[:, 1]
        new_states[:, 1] = states[:, 1] + dt * (1 / m * torch.multiply(torch.sum(action, dim=1), torch.sin(states[:, 4])))
        new_states[:, 2] = states[:, 2] + dt * states[:, 3]
        new_states[:, 3] = states[:, 3] + dt * (1 / m * torch.multiply(torch.sum(action, dim=1), torch.cos(states[:, 4])) - g)
        theta = torch.atan2(states[:, -2], states[:, -3])
        new_theta = theta + dt * states[:, 5]
        new_states[:, 4] = torch.cos(new_theta)
        new_states[:, 5] = torch.sin(new_theta)
        new_states[:, 6] = states[:, 6] + dt * (1 / 2 / Iyy * (action[:, 1] - action[:, 0]))

        # new_states = states + dt * dot_states

        return new_states

    def _get_reward(self, obs, action):
        if self.q_net.dynamics_type == 'pendulum':
            assert obs.shape[1] == 3
            th = torch.atan2(obs[:, 1], obs[:, 0])  # 1 is sin, 0 is cosine
            thdot = obs[:, 2]
            action = torch.reshape(action, (action.shape[0],))
            th = angle_normalize(th)
            reward = -(th ** 2 + 0.1 * thdot ** 2 + 0.01 * action ** 2)
        elif self.q_net.dynamics_type == 'quadrotor_2d':
            if isinstance(self.q_net.dynamics_parameters.get('stabilizing_target'), list):
                stabilizing_target = torch.tensor(self.q_net.dynamics_parameters.get('stabilizing_target'))
            else:
                stabilizing_target = self.q_net.dynamics_parameters.get('stabilizing_target')
            if self.q_net.sin_input is False:
                assert obs.shape[1] == 6
                state_error = obs - stabilizing_target
                reward = torch.exp(-(torch.sum(1. * state_error ** 2, dim=1) + torch.sum(0.0001 * action ** 2, dim=1)))
            else:
                assert obs.shape[1] == 7
                th = torch.unsqueeze(torch.atan2(obs[:, -2], obs[:, -3]), dim=1)  # -2 is sin, -3 is cos
                obs = torch.hstack([obs[:, :4], th, obs[:, -1:]])
                state_error = obs - stabilizing_target
                reward = torch.exp(-(torch.sum(1. * state_error ** 2, dim=1) + torch.sum(0.0001 * action ** 2, dim=1)))
        return torch.reshape(reward, (reward.shape[0], 1))


# ModelCatalog.register_custom_model("sac_rf_model", SACTorchRFModel)

### test model

if __name__ == '__main__':
    obs_space = Box(-1.0, 1.0, (7,))
    action_space = Box(-1.0, 1.0, (2,))

    # Run in eager mode for value checking and debugging.
    # tf1.enable_eager_execution()

    # __sphinx_doc_model_construct_1_begin__
    rf_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=1,
        model_config=MODEL_DEFAULTS,
        framework="torch",
        # Providing the `model_interface` arg will make the factory
        # wrap the chosen default model with our new model API class
        # (DuelingQModel). This way, both `forward` and `get_q_values`
        # are available in the returned class.
        model_interface=SACTorchRFModel,
        name="rf_q_model",
        policy_model_config=SACConfig().policy_model_config,
        q_model_config=RF_MODEL_DEFAULTS
    )
    # __sphinx_doc_model_construct_1_end__

    batch_size = 10
    input_ = np.array([obs_space.sample() for _ in range(batch_size)])
    actions_ = np.array([action_space.sample() for _ in range(batch_size)])
    # Note that for PyTorch, you will have to provide torch tensors here.
    # if args.framework == "torch":
    input_ = torch.from_numpy(input_)
    actions = torch.from_numpy(actions_)

    input_dict = SampleBatch(obs=input_, _is_training=True)
    # actions = SampleBatch(action=actions_, _is_training=False)
    # out, state_outs = rf_model(input_dict=input_dict)
    # print(out, state_outs)
    # assert out.shape == (10, 256)
    # Pass `out` into `get_q_values`
    q_values, _ = rf_model.get_q_values(input_dict, actions)
    action, _ = rf_model.get_action_model_outputs(input_dict)
    print(action)
    print(q_values)
    # assert q_values.shape == (10, action_space.n)

    # # Test API wrapper for single value Q-head from obs/action input.
    #
    # obs_space = Box(-1.0, 1.0, (3,))
    # action_space = Box(-1.0, -1.0, (2,))
    #
    # # __sphinx_doc_model_construct_2_begin__
    # my_cont_action_q_model = ModelCatalog.get_model_v2(
    #     obs_space=obs_space,
    #     action_space=action_space,
    #     num_outputs=2,
    #     model_config=MODEL_DEFAULTS,
    #     framework='torch',
    #     # Providing the `model_interface` arg will make the factory
    #     # wrap the chosen default model with our new model API class
    #     # (DuelingQModel). This way, both `forward` and `get_q_values`
    #     # are available in the returned class.
    #     model_interface=ContActionQModel
    #     if args.framework != "torch"
    #     else TorchContActionQModel,
    #     name="cont_action_q_model",
    # )
    # # __sphinx_doc_model_construct_2_end__
    #
    # batch_size = 10
    # input_ = np.array([obs_space.sample() for _ in range(batch_size)])
    #
    # # Note that for PyTorch, you will have to provide torch tensors here.
    # if args.framework == "torch":
    #     input_ = torch.from_numpy(input_)
    #
    # input_dict = SampleBatch(obs=input_, _is_training=False)
    # # Note that for PyTorch, you will have to provide torch tensors here.
    # out, state_outs = my_cont_action_q_model(input_dict=input_dict)
    # assert out.shape == (10, 256)
    # # Pass `out` and an action into `my_cont_action_q_model`
    # action = np.array([action_space.sample() for _ in range(batch_size)])
    # if args.framework == "torch":
    #     action = torch.from_numpy(action)
    #
    # q_value = my_cont_action_q_model.get_single_q_value(out, action)
    # assert q_value.shape == (10, 1)
