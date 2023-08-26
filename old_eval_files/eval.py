from ray.rllib.algorithms.sac import SACConfig, sac, RFSACConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env

import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make, register
from gymnasium.wrappers import TransformReward
from ray.rllib.scripts import evaluate
from ray.rllib.algorithms.sac import SACConfig
from copy import deepcopy
from main_tune import env_creator_cartpole

ModelCatalog.register_custom_model("sac_rf_model", SACTorchRFModel)

REWARD_SCALE = 0.05

# RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 65536,
#                                       'sigma': 0,
#                                       'learn_rf': False,
#                                       'dynamics_type': 'quadrotor_2d', # pendulum, quadrotor_2d
#                                       'dynamics_parameters': {'stabilizing_target': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
#                                                               'reward_exponential': False,
#                                                               'reward_scale': REWARD_SCALE,
#                                                               }}
RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 2048,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'CartPoleContinuous', # Pendulum, Quadrotor2D
                                      'sin_input': False}

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)


def env_creator(env_config):
    CONFIG_FACTORY = ConfigFactory()
    CONFIG_FACTORY.parser.set_defaults(overrides=['./quad_2d_env_config/stabilization_eval.yaml'])
    config = CONFIG_FACTORY.merge()
    env = make('quadrotor', **config.quadrotor_config)
    return TransformReward(env, lambda r: REWARD_SCALE * r)

register_env('Quadrotor-v1', env_creator)
register_env('CartPoleContinuous-v0', env_creator_cartpole)
config = RFSACConfig().update_from_dict({'q_model_config': RF_MODEL_DEFAULTS, 'evaluation_config': {}})

# config = SACConfig().environment(env='Quadrotor-v1')\
#     .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=0).evaluation(
#     evaluation_interval=1, evaluation_num_workers=0)
#
# # config.create_env_on_local_worker = True
#
# algo = config.build()
#
# algo.restore()
#
# result = algo.evaluate()
#
# print(pretty_print(result))

evaluate(
    # checkpoint='/home/mht/ray_results/SAC_Pendulum-v1_2023-04-23_19-18-33qzefa_7_/checkpoint_000996',
    checkpoint='/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-21lnb8dai3/checkpoint_000151',
    algo= 'RFSAC',
    env= 'CartPoleContinuous-v0',
    config=config,
)

