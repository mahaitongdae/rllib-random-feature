from ray.rllib.algorithms.sac import SACConfig, sac
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env

import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make, register

from copy import deepcopy

ModelCatalog.register_custom_model("sac_rf_model", SACTorchRFModel)

RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 256,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'quadrotor_2d', # pendulum, quadrotor_2d
                                      'dynamics_parameters': {'stabilizing_target': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                                              'reward_exponential': False,
                                                              }}

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)

CONFIG_FACTORY = ConfigFactory()
CONFIG_FACTORY.parser.set_defaults(overrides=['../quad_2d_env_config/stabilization.yaml'])
config = CONFIG_FACTORY.merge()
env = make('quadrotor', **config.quadrotor_config)

env.reset()
action = env.action_space.sample()
env.step(action)
