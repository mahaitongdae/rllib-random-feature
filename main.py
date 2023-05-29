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
                                      'dynamics_type': 'Quadrotor2D', # Pendulum, Quadrotor2D
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

def env_creator_cartpole(env_config):
    from gymnasium.envs.registration import register

    register(id='CartPoleContinuous-v0',
             entry_point='envs:CartPoleEnv',
             max_episode_steps=300)
    env = gymnasium.make('CartPoleContinuous-v0', render_mode='human')
    return env

def train_rfsac(args):
    RF_MODEL_DEFAULTS.update({'random_feature_dim': args.random_feature_dim})
    RF_MODEL_DEFAULTS.update({'dynamics_type' : args.env_id.split('-')[0]})

    register_env('Quadrotor2D-v1', env_creator)
    register_env('CartPoleContinuous-v0', env_creator_cartpole)

    if args.algo == 'RFSAC':
        config = RFSACConfig().environment(env=args.env_id)\
            .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=2)

    elif args.algo == 'SAC':
        config = SACConfig().environment(env=args.env_id)\
            .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=2)

    if args.eval:
        config = config.evaluation(
            evaluation_interval=1,
            evaluation_duration=5,
            evaluation_num_workers=1,
            evaluation_config=RFSACConfig.overrides(render_env=True)
            )

    algo = config.build()

    # algo.restore('/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-215bpvmwd3/checkpoint_000451')
    algo.restore('/home/mht/ray_results/SAC_CartPoleContinuous-v0_2023-05-29_06-37-2148n_4kxq/checkpoint_000451')

    train_iter = 1 if args.eval else 500
    for i in range(train_iter):
        result = algo.train()
        print(pretty_print(result))

        if i % 50 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_feature_dim", default=4096, type=int)
    parser.add_argument("--env_id", default='CartPoleContinuous-v0', type=str)
    parser.add_argument("--algo", default='SAC', type=str)
    parser.add_argument("--eval", default=False, type=bool)
    args = parser.parse_args()
    train_rfsac(args)
    # env = env_creator_cartpole(ENV_CONFIG)
    # print(env.reset())
    # print(env.observation_space)
    # print(env.action_space)
    # action = env.action_space.sample()
    # print(env.step(action))
    # print(env.step(action))
