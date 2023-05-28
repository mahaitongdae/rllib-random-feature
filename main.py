import gymnasium
from ray.rllib.algorithms.sac import SACConfig, sac, RFSACConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env

import sys
sys.path.append('/home/mht/PycharmProjects/safe-control-gym')
import safe_control_gym
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make, register
from gymnasium.wrappers import TransformReward
import argparse
import numpy as np

from copy import deepcopy

ModelCatalog.register_custom_model("sac_rf_model", SACTorchRFModel)

REWARD_SCALE = 0.1

RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 8192,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'quadrotor_2d', # pendulum, quadrotor_2d
                                      'dynamics_parameters': {'stabilizing_target': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                                              'reward_exponential': True,
                                                              'reward_scale': REWARD_SCALE,
                                                              }}

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)

ENV_CONFIG = {'sin_input': False}

RF_MODEL_DEFAULTS.update(ENV_CONFIG)

class TransformTriangleObservationWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        low = np.array([
            -env.x_threshold, -np.finfo(np.float32).max,
            env.GROUND_PLANE_Z, -np.finfo(np.float32).max,
            -1., -1., -np.finfo(np.float32).max
        ])
        high = np.array([
            env.x_threshold, np.finfo(np.float32).max,
            env.z_threshold, np.finfo(np.float32).max,
            1., 1., np.finfo(np.float32).max
        ])
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        theta = observation[-2]
        sin_cos_theta = np.array([np.cos(theta), np.sin(theta)])
        theta_dot = observation[-1:]
        return np.hstack([observation[:4], sin_cos_theta, theta_dot])

def env_creator(env_config):
    CONFIG_FACTORY = ConfigFactory()
    CONFIG_FACTORY.parser.set_defaults(overrides=['./quad_2d_env_config/stabilization.yaml'])
    config = CONFIG_FACTORY.merge()
    env = make('quadrotor', **config.quadrotor_config)
    if ENV_CONFIG.get('sin_input'):
        trans_rew_env = TransformReward(env, lambda r: REWARD_SCALE * r)
        return TransformTriangleObservationWrapper(trans_rew_env)
    else:
        return TransformReward(env, lambda r: REWARD_SCALE * r)

def env_creator_mujoco_cartpole(env_config):
    from gymnasium.envs.registration import register

    register(id='CartPole-v2',
             entry_point='envs:CartPoleEnv')
    env = gymnasium.make('CartPole-v2')
    return env

def train_rfsac(args):
    RF_MODEL_DEFAULTS.update({'random_feature_dim': args.random_feature_dim})

    register_env('Quadrotor-v1', env_creator)

    config = RFSACConfig().environment(env='Quadrotor-v1')\
        .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=1)

    algo = config.build()

    algo.restore('/home/mht/ray_results/RFSAC_Quadrotor-v1_2023-05-08_18-58-048lea_yvt/checkpoint_000451')

    for i in range(500):
        result = algo.train()
        print(pretty_print(result))
        print(result.keys())

        if i % 50 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--random_feature_dim", default=32768, type=int)
    # args = parser.parse_args()
    # train_rfsac(args)
    env = env_creator_mujoco_cartpole(ENV_CONFIG)
    print(env.reset())
    print(env.observation_space)
    print(env.action_space)
    action = env.action_space.sample()
    print(env.step(action))
