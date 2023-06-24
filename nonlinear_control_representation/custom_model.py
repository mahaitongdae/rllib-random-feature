import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType

torch, nn = try_import_torch()


class KernelRepresetationModel(TorchModelV2, nn.Module):
    """
    Random feature network.
    """

    ACTION_DIM_DICT = {'Quadrotor2D': 2,
                       'Pendubot': 1,
                       'Pendulum': 1,
                       'CartPoleContinuous': 1}

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.model_config = model_config.get('custom_model_config')

        # Set dynamics-specific property
        self.dynamics_type = self.model_config.get('dynamics_type')
        self.action_space_dim = self.ACTION_DIM_DICT.get(self.dynamics_type)
        if self.dynamics_type == 'quadrotor_2d':
            self.stabilizing_target = self.model_config.get('stabilizing_target')
        self.dynamics_parameters = self.model_config.get('dynamics_parameters')
        self.sin_input = self.model_config.get('sin_input', False)

        self.prev_layer_size = int(np.product(obs_space.shape) - self.action_space_dim)
        self.feature_dim = self.model_config.get('feature_dim')
        self.sigma = self.model_config.get('sigma')

        # self.device = torch.device('cuda' if torch.cuda.is_avaliable() else 'cpu')

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
        new_states[:, 1] = states[:, 1] + dt * (
                    1 / m * torch.multiply(torch.sum(action, dim=1), torch.sin(states[:, 4])))
        new_states[:, 2] = states[:, 2] + dt * states[:, 3]
        new_states[:, 3] = states[:, 3] + dt * (
                    1 / m * torch.multiply(torch.sum(action, dim=1), torch.cos(states[:, 4])) - g)
        theta = torch.atan2(states[:, -2], states[:, -3])
        new_theta = theta + dt * states[:, 5]
        new_states[:, 4] = torch.cos(new_theta)
        new_states[:, 5] = torch.sin(new_theta)
        new_states[:, 6] = states[:, 6] + dt * (1 / 2 / Iyy * (action[:, 1] - action[:, 0]))

        # new_states = states + dt * dot_states

        return new_states

    def cartpole_f_4d(self, states, action, ):
        """

        :param states: # x, x_dot, theta, theta_dot
        :param action: Force applied to the cart
        :return: new states
        """
        masscart = 1.0
        masspole = 0.1
        length = 0.5
        total_mass = masspole + masscart
        polemass_length = masspole * length
        dt = 0.02
        gravity = 9.81
        new_states = torch.empty_like(states)
        new_states[:, 0] = states[:, 0] + dt * states[:, 1]
        new_states[:, 2] = states[:, 2] + dt * states[:, 3]
        theta = states[:, 2]
        theta_dot = states[:, 3]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        force = torch.squeeze(10. * action)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = 1. / total_mass * (
                force + polemass_length * theta_dot ** 2 * sintheta
        )
        thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        new_states[:, 1] = states[:, 1] + dt * xacc
        new_states[:, 3] = theta_dot + dt * thetaacc
        return new_states

    def cartpole_f_5d(self, states, action, ):
        """

        :param states: # x, x_dot, sin_theta, cos_theta, theta_dot
        :param action: Force applied to the cart
        :return: new states
        """
        masscart = 1.0
        masspole = 0.1
        length = 0.5
        total_mass = masspole + masscart
        polemass_length = masspole * length
        dt = 0.02
        gravity = 9.81
        new_states = torch.empty_like(states)
        new_states[:, 0] = states[:, 0] + dt * states[:, 1]
        costheta = states[:, -3]
        sintheta = states[:, -2]
        theta_dot = states[:, -1]
        theta = torch.atan2(sintheta, costheta)
        new_theta = theta + dt * theta_dot
        new_states[:, -3] = torch.cos(new_theta)
        new_states[:, -2] = torch.sin(new_theta)
        # new_states[:, 2] = states[:, 2] + dt * states[:, 3]
        # theta = states[:, 2]

        force = torch.squeeze(10. * action)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = 1. / total_mass * (
                force + polemass_length * theta_dot ** 2 * sintheta
        )
        thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        new_states[:, 1] = states[:, 1] + dt * xacc
        new_states[:, 4] = theta_dot + dt * thetaacc
        return new_states

    def pendubot_f_6d(self, states, action):
        import torch
        dt = 0.05
        new_states = torch.empty_like(states)
        cos_theta1, sin_theta1 = states[:, 0], states[:, 1]
        cos_theta2, sin_theta2 = states[:, 2], states[:, 3]
        theta1_dot, theta2_dot = states[:, 4], states[:, 5]
        theta1 = torch.atan2(sin_theta1, cos_theta1)
        theta2 = torch.atan2(sin_theta2, cos_theta2)
        new_theta1 = theta1 + dt * theta1_dot
        new_theta2 = theta2 + dt * theta2_dot
        new_states[:, 0] = torch.cos(new_theta1)
        new_states[:, 1] = torch.sin(new_theta1)
        new_states[:, 2] = torch.cos(new_theta2)
        new_states[:, 3] = torch.sin(new_theta2)

        d1 = 0.089252
        d2 = 0.027630
        d3 = 0.023502
        d4 = 0.011204
        d5 = 0.002938
        g = 9.81

        self.d4 = d4
        self.d5 = d5

        m11 = d1 + d2 + 2 * d3 * torch.cos(theta2)
        m21 = d2 + d3 * torch.cos(theta2)
        # m12 = d2 + d3 * torch.cos(theta2)
        m22 = d2

        mass_matrix = torch.empty((states.shape[0], 2, 2))
        mass_matrix[:, 0, 0] = m11
        mass_matrix[:, 0, 1] = m21
        mass_matrix[:, 1, 0] = m21
        mass_matrix[:, 1, 1] = m22

        self.mass_matrix = mass_matrix

        # mass_matrix = np.array([[m11, m12],
        #                         [m21, m22]])

        c_matrix = torch.empty((states.shape[0], 2, 2))
        c11 = -1. * d3 * torch.sin(theta2) * theta2_dot
        c12 = -d3 * torch.sin(theta2) * (theta2_dot + theta1_dot)
        c21 = d3 * torch.sin(theta2) * theta1_dot
        c22 = torch.zeros_like(theta1)
        c_matrix[:, 0, 0] = c11
        c_matrix[:, 0, 1] = c12
        c_matrix[:, 1, 0] = c21
        c_matrix[:, 1, 1] = c22

        g1 = d4 * torch.cos(theta2) * g + d5 * g * torch.cos(theta1 + theta2)
        g2 = d5 * torch.cos(theta1 + theta2) * g

        g_vec = torch.empty((states.shape[0], 2, 1))
        g_vec[:, 0, 0] = g1
        g_vec[:, 1, 0] = g2

        action = torch.hstack([action, torch.zeros_like(action)])[:, :, np.newaxis]
        acc = torch.linalg.solve(mass_matrix, action - torch.matmul(c_matrix, states[:, -2:][:, :, np.newaxis]) - g_vec)
        new_states[:, 4] = theta1_dot + dt * torch.squeeze(acc[:, 0])
        new_states[:, 5] = theta2_dot + dt * torch.squeeze(acc[:, 1])

        return new_states

    def _get_energy_error(self, obs, action, ke=1.5):
        assert self.dynamics_type == 'Pendubot'
        dot_theta = obs[:, -2:][:, :, np.newaxis]  # batch, 2, 1
        dot_theta_t = obs[:, -2:][:, np.newaxis]  # batch, 1, 2
        cos_theta1, sin_theta1 = obs[:, 0], obs[:, 1]
        cos_theta2, sin_theta2 = obs[:, 2], obs[:, 3]
        sin_theta1_plus_theta2 = torch.multiply(sin_theta1, cos_theta2) + torch.multiply(cos_theta1, sin_theta2)

        kinetic_energy = torch.squeeze(torch.matmul(torch.matmul(dot_theta_t, self.mass_matrix), dot_theta))
        potential_energy = self.d4 * 9.81 * sin_theta1 + self.d5 * 9.81 * sin_theta1_plus_theta2
        energy_on_top = (self.d4 + self.d5) * 9.81
        energy_error = kinetic_energy + potential_energy - energy_on_top

        return ke * energy_error ** 2

    def angle_normalize(self, th):
        return ((th + np.pi) % (2 * np.pi)) - np.pi

    def _get_reward(self, obs, action):
        if self.dynamics_type == 'Pendulum':
            assert obs.shape[1] == 3
            th = torch.atan2(obs[:, 1], obs[:, 0])  # 1 is sin, 0 is cosine
            thdot = obs[:, 2]
            action = torch.reshape(action, (action.shape[0],))
            th = self.angle_normalize(th)
            reward = -(th ** 2 + 0.1 * thdot ** 2 + 0.01 * action ** 2)

        elif self.dynamics_type == 'Quadrotor2D':
            if isinstance(self.dynamics_parameters.get('stabilizing_target'), list):
                stabilizing_target = torch.tensor(self.dynamics_parameters.get('stabilizing_target'))
            else:
                stabilizing_target = self.dynamics_parameters.get('stabilizing_target')
            if self.sin_input is False:
                assert obs.shape[1] == 6
                state_error = obs - stabilizing_target
                reward = -(torch.sum(1. * state_error ** 2, dim=1) + torch.sum(0.0001 * action ** 2, dim=1))
                # if self.model_config.get('reward_exponential'):
                #     reward = torch.exp(reward)
            else:
                assert obs.shape[1] == 7
                th = torch.unsqueeze(torch.atan2(obs[:, -2], obs[:, -3]), dim=1)  # -2 is sin, -3 is cos
                obs = torch.hstack([obs[:, :4], th, obs[:, -1:]])
                state_error = obs - stabilizing_target
                reward = -(torch.sum(1. * state_error ** 2, dim=1) + torch.sum(0.0001 * action ** 2, dim=1))

        elif self.dynamics_type == 'CartPoleContinuous':
            if self.sin_input is False:
                reward = -(torch.sum(obs ** 2, dim=1) + torch.sum(0.01 * action ** 2, dim=1))
            else:
                assert obs.shape[1] == 5
                th = torch.unsqueeze(torch.atan2(obs[:, -2], obs[:, -3]), dim=1)  # -2 is sin, -3 is cos
                obs = torch.hstack([obs[:, :-3], th, obs[:, -1:]])
                reward = -(torch.sum(obs ** 2, dim=1) + torch.sum(0.01 * action ** 2, dim=1))

        elif self.dynamics_type == 'Pendubot':
            if self.sin_input:
                assert obs.shape[1] == 6
                th1dot = obs[:, 4]
                th2dot = obs[:, 5]
                if self.model_config.get('reward_type') == 'lqr':
                    if self.model_config.get('theta_cal') == 'arctan':
                        th1 = torch.atan2(obs[:, 1], obs[:, 0])
                        th2 = torch.atan2(obs[:, 3], obs[:, 2])
                        reward = -1. * ((th1 - np.pi / 2) ** 2 + th1dot ** 2 +
                                        0.01 * th2 ** 2 + 0.01 * th2dot ** 2 + 0.01 * torch.squeeze(action) ** 2)
                    elif self.model_config.get('theta_cal') == 'sin_cos':
                        cos_th1 = obs[:, 0]
                        sin_th1 = obs[:, 1]
                        cos_th2 = obs[:, 2]
                        sin_th2 = obs[:, 3]
                        reward = -1. * ((cos_th1) ** 2 + (sin_th1 - 1.) ** 2 + th1dot ** 2 +
                                        0.01 * (sin_th2) ** 2 + 0.01 * (cos_th2 - 1.) ** 2 +
                                        0.01 * th2dot ** 2 + 0.01 * torch.squeeze(action) ** 2)
                    else:
                        raise NotImplementedError
                elif self.model_config.get('reward_type') == 'energy':
                    if self.model_config.get('theta_cal') == 'arctan':
                        th1 = torch.atan2(obs[:, 1], obs[:, 0])
                        th2 = torch.atan2(obs[:, 3], obs[:, 2])
                        reward = -1. * ((th1 - np.pi / 2) ** 2 + th1dot ** 2 + self._get_energy_error(obs, action))
                    elif self.model_config.get('theta_cal') == 'sin_cos':
                        cos_th1 = obs[:, 0]
                        sin_th1 = obs[:, 1]
                        cos_th2 = obs[:, 2]
                        sin_th2 = obs[:, 3]
                        reward = -1. * ((cos_th1) ** 2 + (sin_th1 - 1.) ** 2 + th1dot ** 2 + self._get_energy_error(obs, action))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                reward_scale = self.model_config.get('reward_scale')
                reward = reward_scale * reward
        # exponent
        if self.model_config.get('reward_exponential'):
            reward = torch.exp(reward)
        return torch.reshape(reward, (reward.shape[0], 1))

    def dynamics(self, obs, actions):
        if self.dynamics_type == 'Quadrotor2D':
            if self.sin_input:
                obs_tp1 = self.quadrotor_f_star_7d(obs, actions)
            else:
                obs_tp1 = self.quadrotor_f_star_6d(obs, actions)
        elif self.dynamics_type == 'Pendulum':
            obs_tp1 = self.pendulum_3d(obs, actions)
        elif self.dynamics_type == 'CartPoleContinuous':
            if self.sin_input is False:
                obs_tp1 = self.cartpole_f_4d(obs, actions)
            else:
                obs_tp1 = self.cartpole_f_5d(obs, actions)
        elif self.dynamics_type == 'Pendubot':
            if self.sin_input:
                obs_tp1 = self.pendubot_f_6d(obs, actions)
            else:
                raise NotImplementedError

        return obs_tp1

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs = input_dict["obs_flat"].float()[:, :-self.action_space_dim]
        actions = input_dict["obs_flat"].float()[:, -self.action_space_dim:]
        obs_tp1 = self.dynamics(obs, actions)

        # self._last_flat_in = obs.reshape(obs_tp1.shape[0], -1)
        # x = torch.cos(self.fournier_random_feature(self._last_flat_in))
        # logits = self.linear1(x)

        reward = self._get_reward(obs, actions)
        logits = torch.zeros_like(reward)
        return reward + logits, state

    def value_function(self) -> TensorType:
        raise NotImplementedError

class RandomFeatureQModel(KernelRepresetationModel, TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        KernelRepresetationModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # initialize Fournier random feature
        fourier_random_feature = nn.Linear(self.prev_layer_size, self.feature_dim)
        if self.sigma > 0:
            nn.init.normal_(fourier_random_feature.weight, std=1. / self.sigma)
        else:
            nn.init.normal_(fourier_random_feature.weight)
        nn.init.uniform_(fourier_random_feature.bias, 0, 2. * np.pi)
        learn_rf = self.model_config.get('learn_rf', False)
        fourier_random_feature.weight.requires_grad = learn_rf
        fourier_random_feature.bias.requires_grad = learn_rf
        self.fournier_random_feature = fourier_random_feature

        linear1 = nn.Linear(self.feature_dim, 1)
        linear1.bias.requires_grad = False
        self.linear1 = linear1


    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()[:, :-self.action_space_dim]
        actions = input_dict["obs_flat"].float()[:, -self.action_space_dim:]
        obs_tp1 = self.dynamics(obs, actions)

        self._last_flat_in = obs.reshape(obs_tp1.shape[0], -1)
        x = torch.cos(self.fournier_random_feature(self._last_flat_in))
        logits = self.linear1(x)
        reward = self._get_reward(obs, actions)
        return reward + logits, state

class NystromSampleQModel(KernelRepresetationModel, TorchModelV2, nn.Module):

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        KernelRepresetationModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        s_low = self.model_config.get('obs_space_low')
        s_high = self.model_config.get('obs_space_high')
        s_dim = self.model_config.get('obs_space_dim')
        s_dim = s_dim[0] if isinstance(s_dim, tuple) else s_dim

        self.nystrom_samples1 = np.random.uniform(s_low, s_high, size=(self.feature_dim, s_dim))

        if self.sigma > 0.0:
            self.kernel = lambda z: np.exp(-np.linalg.norm(z)**2/(2.* self.sigma**2))
        else:
            self.kernel = lambda z: np.exp(-np.linalg.norm(z)**2/(2.))

        K_m1 = self.get_kernel_matrix(self.nystrom_samples1)
        [eig_vals1, S1] = np.linalg.eig(K_m1)  # numpy linalg eig doesn't produce negative eigenvalues... (unlike torch)
        self.eig_vals1 = torch.from_numpy(eig_vals1).float()
        self.S1 = torch.from_numpy(S1).float()
        self.nystrom_samples1 = torch.from_numpy(self.nystrom_samples1)

        self.n_neurons = self.feature_dim
        layer1 = nn.Linear(self.n_neurons, 1)  # try default scaling
        torch.nn.init.zeros_(layer1.bias)
        layer1.bias.requires_grad = False  # weight is the only thing we update
        self.output1 = layer1

    def get_kernel_matrix(self, samples):

        m, d = samples.shape
        K_m = np.empty((m, m))
        for i in np.arange(m):
            for j in np.arange(m):
                K_m[i, j] = self.kernel(samples[i, :] - samples[j, :])
        return K_m

    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs = input_dict["obs_flat"].float()[:, :-self.action_space_dim]
        actions = input_dict["obs_flat"].float()[:, -self.action_space_dim:]
        obs_tp1 = self.dynamics(obs, actions)

        # self._last_flat_in = obs.reshape(obs_tp1.shape[0], -1)
        # x = torch.cos(self.fournier_random_feature(self._last_flat_in))
        # logits = self.linear1(x)

        x1 = self.nystrom_samples1.unsqueeze(0) - obs_tp1.unsqueeze(1)
        K_x1 = torch.exp(-torch.linalg.norm(x1, axis=2) ** 2 / 2).float()
        phi_all1 = (K_x1 @ (self.S1)) @ torch.diag(self.eig_vals1 ** (-0.5))
        phi_all1 = phi_all1 * self.n_neurons * 5 #todo: the scaling matters?
        phi_all1 = phi_all1.to(torch.float32)

        reward = self._get_reward(obs, actions)
        logits = self.output1(phi_all1)
        return reward + logits, state



if __name__ == '__main__':
    from ray.rllib.models.catalog import ModelCatalog, MODEL_DEFAULTS
    from ray.rllib.algorithms.sac import SACConfig
    from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
    from ray.rllib.policy.sample_batch import SampleBatch
    from main import ENV_CONFIG

    ModelCatalog.register_custom_model('random_feature_q', RandomFeatureQModel)
    ModelCatalog.register_custom_model('nystrom_q', NystromSampleQModel)

    obs_space = Box(-1.0, 1.0, (6,))
    action_space = Box(-1.0, 1.0, (1,))

    RF_MODEL_DEFAULTS: ModelConfigDict = {
                                          'custom_model': 'nystrom_q',
                                          'custom_model_config': {
                                              'feature_dim': 256,
                                              'sigma': 0,
                                              'learn_rf': False,
                                              'dynamics_type': 'Pendubot',
                                              'sin_input': True,
                                              'obs_space_high': obs_space.high,
                                              'obs_space_low': obs_space.low,
                                              'obs_space_dim': obs_space.shape,
                                              'dynamics_parameters': {
                                                  'stabilizing_target': torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
                                              },

                                         }
                                     }
    RF_MODEL_DEFAULTS.get('custom_model_config').update(ENV_CONFIG)

    # RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)


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
        model_interface=SACTorchModel,
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
