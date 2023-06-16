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
        # x = torch.div(x, 1. / self.random_feature_dim)
        logits = self.linear1(x)
        return logits, state

    def value_function(self) -> TensorType:
        raise NotImplementedError


class SACTorchModel(TorchModelV2, nn.Module):
    """Extension of the standard TorchModelV2 for SAC.

    To customize, do one of the following:
    - sub-class SACTorchModel and override one or more of its methods.
    - Use SAC's `q_model_config` and `policy_model` keys to tweak the default model
      behaviors (e.g. fcnet_hiddens, conv_filters, etc..).
    - Use SAC's `q_model_config->custom_model` and `policy_model->custom_model` keys
      to specify your own custom Q-model(s) and policy-models, which will be
      created within this SACTFModel (see `build_policy_model` and
      `build_q_model`.

    Note: It is not recommended to override the `forward` method for SAC. This
    would lead to shared weights (between policy and Q-nets), which will then
    not be optimized by either of the critic- or actor-optimizers!

    Data flow:
        `obs` -> forward() (should stay a noop method!) -> `model_out`
        `model_out` -> get_policy_output() -> pi(actions|obs)
        `model_out`, `actions` -> get_q_values() -> Q(s, a)
        `model_out`, `actions` -> get_twin_q_values() -> Q_twin(s, a)
    """

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
        """Initializes a SACTorchModel instance.
        7
                Args:
                    policy_model_config: The config dict for the
                        policy network.
                    q_model_config: The config dict for the
                        Q-network(s) (2 if twin_q=True).
                    twin_q: Build twin Q networks (Q-net and target) for more
                        stable Q-learning.
                    initial_alpha: The initial value for the to-be-optimized
                        alpha parameter (default: 1.0).
                    target_entropy (Optional[float]): A target entropy value for
                        the to-be-optimized alpha parameter. If None, will use the
                        defaults described in the papers for SAC (and discrete SAC).

                Note that the core layers for forward() are not defined here, this
                only defines the layers for the output heads. Those layers for
                forward() should be defined in subclasses of SACModel.
        """
        nn.Module.__init__(self)
        super(SACTorchModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        if isinstance(action_space, Discrete):
            self.action_dim = action_space.n
            self.discrete = True
            action_outs = q_outs = self.action_dim
        elif isinstance(action_space, Box):
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            action_outs = 2 * self.action_dim
            q_outs = 1
        else:
            assert isinstance(action_space, Simplex)
            self.action_dim = np.product(action_space.shape)
            self.discrete = False
            action_outs = self.action_dim
            q_outs = 1

        # Build the policy network.
        self.action_model = self.build_policy_model(
            self.obs_space, action_outs, policy_model_config, "policy_model"
        )

        # Build the Q-network(s).
        self.q_net = self.build_q_model(
            self.obs_space, self.action_space, q_outs, q_model_config, "q"
        )
        if twin_q:
            self.twin_q_net = self.build_q_model(
                self.obs_space, self.action_space, q_outs, q_model_config, "twin_q"
            )
        else:
            self.twin_q_net = None

        log_alpha = nn.Parameter(
            torch.from_numpy(np.array([np.log(initial_alpha)])).float()
        )
        self.register_parameter("log_alpha", log_alpha)

        # Auto-calculate the target entropy.
        if target_entropy is None or target_entropy == "auto":
            # See hyperparams in [2] (README.md).
            if self.discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / action_space.n), dtype=np.float32
                )
            # See [1] (README.md).
            else:
                target_entropy = -np.prod(action_space.shape)

        target_entropy = nn.Parameter(
            torch.from_numpy(np.array([target_entropy])).float(), requires_grad=False
        )
        self.register_parameter("target_entropy", target_entropy)

    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        """The common (Q-net and policy-net) forward pass.

        NOTE: It is not(!) recommended to override this method as it would
        introduce a shared pre-network, which would be updated by both
        actor- and critic optimizers.
        """
        return input_dict["obs"], state

    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        """Builds the policy model used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own policy net. Alternatively, simply set `custom_model` within the
        top level SAC `policy_model` config key to make this default
        implementation of `build_policy_model` use your custom policy network.

        Returns:
            TorchModelV2: The TorchModelV2 policy sub-model.
        """
        model = ModelCatalog.get_model_v2(
            obs_space,
            self.action_space,
            num_outputs,
            policy_model_config,
            framework="torch",
            name=name,
        )
        return model

    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
        """Builds one of the (twin) Q-nets used by this SAC.

        Override this method in a sub-class of SACTFModel to implement your
        own Q-nets. Alternatively, simply set `custom_model` within the
        top level SAC `q_model_config` config key to make this default implementation
        of `build_q_model` use your custom Q-nets.

        Returns:
            TorchModelV2: The TorchModelV2 Q-net sub-model.
        """
        self.concat_obs_and_actions = False
        if self.discrete:
            input_space = obs_space
        else:
            orig_space = getattr(obs_space, "original_space", obs_space)
            if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
                input_space = Box(
                    float("-inf"),
                    float("inf"),
                    shape=(orig_space.shape[0] + action_space.shape[0],),
                )
                self.concat_obs_and_actions = True
            else:
                input_space = gym.spaces.Tuple([orig_space, action_space])

        model = ModelCatalog.get_model_v2(
            input_space,
            action_space,
            num_outputs,
            q_model_config,
            framework="torch",
            name=name,
        )
        return model

    def get_q_values(
            self, model_out: TensorType, actions: Optional[TensorType] = None
    ) -> TensorType:
        """Returns Q-values, given the output of self.__call__().

        This implements Q(s, a) -> [single Q-value] for the continuous case and
        Q(s) -> [Q-values for all actions] for the discrete case.

        Args:
            model_out: Feature outputs from the model layers
                (result of doing `self.__call__(obs)`).
            actions (Optional[TensorType]): Continuous action batch to return
                Q-values for. Shape: [BATCH_SIZE, action_dim]. If None
                (discrete action case), return Q-values for all actions.

        Returns:
            TensorType: Q-values tensor of shape [BATCH_SIZE, 1].
        """
        return self._get_q_value(model_out, actions, self.q_net)

    def get_twin_q_values(
            self, model_out: TensorType, actions: Optional[TensorType] = None
    ) -> TensorType:
        """Same as get_q_values but using the twin Q net.

        This implements the twin Q(s, a).

        Args:
            model_out: Feature outputs from the model layers
                (result of doing `self.__call__(obs)`).
            actions (Optional[Tensor]): Actions to return the Q-values for.
                Shape: [BATCH_SIZE, action_dim]. If None (discrete action
                case), return Q-values for all actions.

        Returns:
            TensorType: Q-values tensor of shape [BATCH_SIZE, 1].
        """
        return self._get_q_value(model_out, actions, self.twin_q_net)

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
            if self.concat_obs_and_actions:
                input_dict = {"obs": torch.cat([model_out, actions], dim=-1)}
            else:
                # TODO(junogng) : SampleBatch doesn't support list columns yet.
                #     Use ModelInputDict.
                input_dict = {"obs": (model_out, actions)}
        # Discrete case -> return q-vals for all actions.
        else:
            input_dict = {"obs": model_out}
        # Switch on training mode (when getting Q-values, we are usually in
        # training).
        input_dict["is_training"] = True

        return net(input_dict, [], None)

    def get_action_model_outputs(
            self,
            model_out: TensorType,
            state_in: List[TensorType] = None,
            seq_lens: TensorType = None,
    ) -> (TensorType, List[TensorType]):
        """Returns distribution inputs and states given the output of
        policy.model().

        For continuous action spaces, these will be the mean/stddev
        distribution inputs for the (SquashedGaussian) action distribution.
        For discrete action spaces, these will be the logits for a categorical
        distribution.

        Args:
            model_out: Feature outputs from the model layers
                (result of doing `model(obs)`).
            state_in List(TensorType): State input for recurrent cells
            seq_lens: Sequence lengths of input- and state
                sequences

        Returns:
            TensorType: Distribution inputs for sampling actions.
        """

        def concat_obs_if_necessary(obs: TensorStructType):
            """Concat model outs if they come as original tuple observations."""
            if isinstance(obs, (list, tuple)):
                obs = torch.cat(obs, dim=-1)
            elif isinstance(obs, dict):
                obs = torch.cat(
                    [
                        torch.unsqueeze(val, 1) if len(val.shape) == 1 else val
                        for val in tree.flatten(obs.values())
                    ],
                    dim=-1,
                )
            return obs

        if state_in is None:
            state_in = []

        if isinstance(model_out, dict) and "obs" in model_out:
            # Model outs may come as original Tuple observations
            if isinstance(self.action_model.obs_space, Box):
                model_out["obs"] = concat_obs_if_necessary(model_out["obs"])
            return self.action_model(model_out, state_in, seq_lens)
        else:
            if isinstance(self.action_model.obs_space, Box):
                model_out = concat_obs_if_necessary(model_out)
            return self.action_model({"obs": model_out}, state_in, seq_lens)

    def policy_variables(self):
        """Return the list of variables for the policy net."""

        return self.action_model.variables()

    def q_variables(self):
        """Return the list of variables for Q / twin Q nets."""

        return self.q_net.variables() + (
            self.twin_q_net.variables() if self.twin_q_net else []
        )


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

        self.q_net = RandomFeatureNetwork(obs_space, action_space, q_outs, q_model_config, 'q')

        if twin_q:
            self.twin_q_net = RandomFeatureNetwork(obs_space, action_space, q_outs, q_model_config, 'twin_q')

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
            if self.q_net.dynamics_type == 'Quadrotor2D':
                if self.q_net.sin_input:
                    obs_tp1 = self.quadrotor_f_star_7d(model_out, actions)
                else:
                    obs_tp1 = self.quadrotor_f_star_6d(model_out, actions)
            elif self.q_net.dynamics_type == 'Pendulum':
                obs_tp1 = self.pendulum_3d(model_out, actions)
            elif self.q_net.dynamics_type == 'CartPoleContinuous':
                if self.q_net.sin_input is False:
                    obs_tp1 = self.cartpole_f_4d(model_out, actions)
                else:
                    obs_tp1 = self.cartpole_f_5d(model_out, actions)
            elif self.q_net.dynamics_type == 'Pendubot':
                if self.q_net.sin_input:
                    obs_tp1 = self.pendubot_f_6d(model_out, actions)
                else:
                    raise NotImplementedError
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
        c11 = -1. * d3 * np.sin(theta2) * theta2_dot
        c12 = -d3 * np.sin(theta2) * (theta2_dot + theta1_dot)
        c21 = d3 * np.sin(theta2) * theta1_dot
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
        assert self.q_net.dynamics_type == 'Pendubot'
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

    def _get_reward(self, obs, action):
        if self.q_net.dynamics_type == 'Pendulum':
            assert obs.shape[1] == 3
            th = torch.atan2(obs[:, 1], obs[:, 0])  # 1 is sin, 0 is cosine
            thdot = obs[:, 2]
            action = torch.reshape(action, (action.shape[0],))
            th = angle_normalize(th)
            reward = -(th ** 2 + 0.1 * thdot ** 2 + 0.01 * action ** 2)

        elif self.q_net.dynamics_type == 'Quadrotor2D':
            if isinstance(self.q_net.dynamics_parameters.get('stabilizing_target'), list):
                stabilizing_target = torch.tensor(self.q_net.dynamics_parameters.get('stabilizing_target'))
            else:
                stabilizing_target = self.q_net.dynamics_parameters.get('stabilizing_target')
            if self.q_net.sin_input is False:
                assert obs.shape[1] == 6
                state_error = obs - stabilizing_target
                reward = -(torch.sum(1. * state_error ** 2, dim=1) + torch.sum(0.0001 * action ** 2, dim=1))
                # if self.q_net.dynamics_parameters.get('reward_exponential'):
                #     reward = torch.exp(reward)
            else:
                assert obs.shape[1] == 7
                th = torch.unsqueeze(torch.atan2(obs[:, -2], obs[:, -3]), dim=1)  # -2 is sin, -3 is cos
                obs = torch.hstack([obs[:, :4], th, obs[:, -1:]])
                state_error = obs - stabilizing_target
                reward = -(torch.sum(1. * state_error ** 2, dim=1) + torch.sum(0.0001 * action ** 2, dim=1))

        elif self.q_net.dynamics_type == 'CartPoleContinuous':
            if self.q_net.sin_input is False:
                reward = -(torch.sum(obs ** 2, dim=1) + torch.sum(0.01 * action ** 2, dim=1))
            else:
                assert obs.shape[1] == 5
                th = torch.unsqueeze(torch.atan2(obs[:, -2], obs[:, -3]), dim=1)  # -2 is sin, -3 is cos
                obs = torch.hstack([obs[:, :-3], th, obs[:, -1:]])
                reward = -(torch.sum(obs ** 2, dim=1) + torch.sum(0.01 * action ** 2, dim=1))

        elif self.q_net.dynamics_type == 'Pendubot':
            if self.q_net.sin_input:
                assert obs.shape[1] == 6
                th1dot = obs[:, 4]
                th2dot = obs[:, 5]
                if self.q_net.dynamics_parameters.get('reward_type') == 'lqr':
                    if self.q_net.dynamics_parameters.get('theta_cal') == 'arctan':
                        th1 = torch.atan2(obs[:, 1], obs[:, 0])
                        th2 = torch.atan2(obs[:, 3], obs[:, 2])
                        reward = -1. * ((th1 - np.pi / 2) ** 2 + th1dot ** 2 +
                                        0.01 * th2 ** 2 + 0.01 * th2dot ** 2 + 0.01 * torch.squeeze(action) ** 2)
                    elif self.q_net.dynamics_parameters.get('theta_cal') == 'sin_cos':
                        cos_th1 = obs[:, 0]
                        sin_th1 = obs[:, 1]
                        cos_th2 = obs[:, 2]
                        sin_th2 = obs[:, 3]
                        reward = -1. * ((cos_th1) ** 2 + (sin_th1 - 1.) ** 2 + th1dot ** 2 +
                                        0.01 * (sin_th2) ** 2 + 0.01 * (cos_th2 - 1.) ** 2 +
                                        0.01 * th2dot ** 2 + 0.01 * torch.squeeze(action) ** 2)
                    else:
                        raise NotImplementedError
                elif self.q_net.dynamics_parameters.get('reward_type') == 'energy':
                    if self.q_net.dynamics_parameters.get('theta_cal') == 'arctan':
                        th1 = torch.atan2(obs[:, 1], obs[:, 0])
                        th2 = torch.atan2(obs[:, 3], obs[:, 2])
                        reward = -1. * ((th1 - np.pi / 2) ** 2 + th1dot ** 2 + self._get_energy_error(obs, action))
                    elif self.q_net.dynamics_parameters.get('theta_cal') == 'sin_cos':
                        cos_th1 = obs[:, 0]
                        sin_th1 = obs[:, 1]
                        cos_th2 = obs[:, 2]
                        sin_th2 = obs[:, 3]
                        reward = -1. * ((cos_th1) ** 2 + (sin_th1 - 1.) ** 2 + th1dot ** 2 + self._get_energy_error(obs, action))
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                reward_scale = self.q_net.dynamics_parameters.get('reward_scale')
                reward = reward_scale * reward
        # exponent
        if self.q_net.dynamics_parameters.get('reward_exponential'):
            reward = torch.exp(reward)
        return torch.reshape(reward, (reward.shape[0], 1))


if __name__ == '__main__':
    from ray.rllib.models.catalog import ModelCatalog, MODEL_DEFAULTS
    from ray.rllib.algorithms.sac import SACConfig
    from ray.rllib.policy.sample_batch import SampleBatch

    RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 256,
                                          'sigma': 0,
                                          'learn_rf': False,
                                          'dynamics_type': 'Pendubot',
                                          'dynamics_parameters': {
                                              'stabilizing_target': torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
                                              'reward_scale': 10.,
                                              'reward_exponential': False,
                                              'reward_type': 'lqr',
                                          },
                                          'sin_input': True}

    RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)
    obs_space = Box(-1.0, 1.0, (6,))
    action_space = Box(-1.0, 1.0, (1,))

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
