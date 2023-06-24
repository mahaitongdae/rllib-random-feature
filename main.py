import gymnasium
import ray
from ray.rllib.algorithms.sac import SACConfig, sac, RFSACConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env, ENV_CREATOR, _global_registry

import os.path as osp
import sys
sys.path.append('/home/mht/PycharmProjects/safe-control-gym')
try:
    import safe_control_gym
    from safe_control_gym.utils.configuration import ConfigFactory
    from safe_control_gym.utils.registration import make, register
except:
    pass
from gymnasium.wrappers import TransformReward
import argparse
import numpy as np
import json
import copy
from custom_model import RandomFeatureQModel, NystromSampleQModel

from copy import deepcopy

ModelCatalog.register_custom_model('random_feature_q', RandomFeatureQModel)
ModelCatalog.register_custom_model('nystrom_q', NystromSampleQModel)

RF_MODEL_DEFAULTS: ModelConfigDict = {
                                          'custom_model': 'nystrom_q',
                                          'custom_model_config': {
                                              'feature_dim': 8192,
                                              'sigma': 0,
                                              'learn_rf': False,
                                              'dynamics_type': 'Pendubot',
                                              'obs_space_high': None,
                                              'obs_space_low': None,
                                              'obs_space_dim': None,
                                              'dynamics_parameters': {
                                                  # 'stabilizing_target': torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
                                                  'reward_scale': 10.,
                                                  'reward_exponential': False,
                                                  'reward_type': 'lqr',
                                              },

                                         }
                                     }


ENV_CONFIG = {'sin_input': True,
              'reward_exponential': False,
              'reward_scale': 10.,
              'reward_type' : 'energy',
              'theta_cal': 'sin_cos',
              'noisy': False,
              'noise_scale': 0.
              }

custom_model_config = RF_MODEL_DEFAULTS.get('custom_model_config')
custom_model_config.update(ENV_CONFIG)

class TransformTriangleObservationWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # low = np.array([
        #     -env.x_threshold, -np.finfo(np.float32).max,
        #     env.GROUND_PLANE_Z, -np.finfo(np.float32).max,
        #     -1., -1., -np.finfo(np.float32).max
        # ])
        # high = np.array([
        #     env.x_threshold, np.finfo(np.float32).max,
        #     env.z_threshold, np.finfo(np.float32).max,
        #     1., 1., np.finfo(np.float32).max
        # ])
        low = env.observation_space.low
        high = env.observation_space.high
        transformed_low = np.hstack([low[:-2], [-1., -1.,], low[-1:]])
        transformed_high = np.hstack([high[:-2], [1., 1.,], high[-1:]])

        self.observation_space = gymnasium.spaces.Box(low=transformed_low, high=transformed_high, dtype=np.float32)

class TransformDoubleTriangleObservationWrapper(gymnasium.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # low = np.array([
        #     -env.x_threshold, -np.finfo(np.float32).max,
        #     env.GROUND_PLANE_Z, -np.finfo(np.float32).max,
        #     -1., -1., -np.finfo(np.float32).max
        # ])
        # high = np.array([
        #     env.x_threshold, np.finfo(np.float32).max,
        #     env.z_threshold, np.finfo(np.float32).max,
        #     1., 1., np.finfo(np.float32).max
        # ])
        low = env.observation_space.low
        high = env.observation_space.high
        transformed_low = np.hstack([[-1., -1., -1., -1.,], low[-2:]])
        transformed_high = np.hstack([[1., 1., 1., 1., ], high[-2:]])

        self.observation_space = gymnasium.spaces.Box(low=transformed_low, high=transformed_high, dtype=np.float32)

    def observation(self, observation):
        theta1 = observation[0]
        theta2 = observation[1]
        sin_cos_theta1 = np.array([np.cos(theta1), np.sin(theta1)])
        sin_cos_theta2 = np.array([np.cos(theta2), np.sin(theta2)])
        theta_dot = observation[-2:]
        return np.hstack([sin_cos_theta1, sin_cos_theta2, theta_dot])

def env_creator(env_config):
    CONFIG_FACTORY = ConfigFactory()
    CONFIG_FACTORY.parser.set_defaults(overrides=['./quad_2d_env_config/stabilization.yaml'])
    config = CONFIG_FACTORY.merge()
    env = make('quadrotor', **config.quadrotor_config)
    if env_config.get('sin_input'):
        trans_rew_env = TransformReward(env, lambda r: env_config.get('reward_scale') * r)
        return TransformTriangleObservationWrapper(trans_rew_env)
    else:
        return TransformReward(env, lambda r: env_config.get('reward_scale') * r)

def env_creator_cartpole(env_config):
    from gymnasium.envs.registration import register

    register(id='CartPoleContinuous-v0',
             entry_point='envs:CartPoleEnv',
             max_episode_steps=300)
    env = gymnasium.make('CartPoleContinuous-v0') #, render_mode='human'
    if env_config.get('reward_exponential'):
        env = TransformReward(env, lambda r: np.exp(r))
    if env_config.get('sin_input'):
        return TransformTriangleObservationWrapper(env)
    else:
        return env

def env_creator_pendubot(env_config):
    from gymnasium.envs.registration import register
    reward_scale_pendubot = env_config.get('reward_scale')
    noisy = env_config.get('noisy')
    noise_scale = env_config.get('noise_scale')
    register(id='Pendubot-v0',
             entry_point='envs:PendubotEnv',
             max_episode_steps=200)
    env = gymnasium.make('Pendubot-v0',
                         noisy=noisy,
                         noisy_scale=noise_scale,
                         reward_type=env_config.get('reward_type'),
                         theta_cal=env_config.get('theta_cal')
                         ) #, render_mode='human'
    env = TransformReward(env, lambda r: reward_scale_pendubot * r)
    if env_config.get('reward_exponential'):
        env = TransformReward(env, lambda r: np.exp(r))
    if env_config.get('sin_input'):
        return TransformDoubleTriangleObservationWrapper(env)
    else:
        return env

def train_rfsac(args):
    ray.init(num_cpus=4, local_mode=True)


    register_env('Quadrotor2D-v1', env_creator)
    register_env('CartPoleContinuous-v0', env_creator_cartpole)
    register_env('Pendubot-v0', env_creator_pendubot)

    # update parameters
    custom_model_config.update({'feature_dim': args.feature_dim})
    custom_model_config.update({'dynamics_type': args.env_id.split('-')[0]})
    ENV_CONFIG.update({
        'reward_exponential': args.reward_exp,
        'reward_type': args.reward_type,
        'reward_scale': args.reward_scale,
        'theta_cal': args.theta_cal
    })
    custom_model_config.update(ENV_CONFIG)
    RF_MODEL_DEFAULTS.update({'comments': args.comments})

    env_creator_func = _global_registry.get(ENV_CREATOR, args.env_id) # from algorithm.py line 2212, Algorithm.__init__()
    env = env_creator_func(ENV_CONFIG)
    custom_model_config.update({
        'obs_space_high': env.observation_space.high.tolist(),
        'obs_space_low': env.observation_space.low.tolist(),
        'obs_space_dim': env.observation_space.shape,
    })
    del env

    # env = ray.tune.

    # if args.algo == 'RFSAC':
    #     config = RFSACConfig().environment(env=args.env_id, env_config=ENV_CONFIG)\
    #         .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=16) #
    #
    # elif args.algo == 'SAC':
    config = SACConfig().environment(env=args.env_id, env_config=ENV_CONFIG)\
        .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=4)

    # if args.eval:
    eval_env_config = copy.deepcopy(ENV_CONFIG)
    eval_env_config.update({
                            'sin_input': True,
                            'reward_exponential': False,
                            'reward_scale': 1.,
                            'reward_type': 'energy',
                            })
    config = config.evaluation(
        # evaluation_parallel_to_training=True,
        evaluation_interval=10,
        evaluation_duration=10,
        evaluation_num_workers=1,
        evaluation_config=RFSACConfig.overrides(render_env=False,
                                                env_config = eval_env_config
                                                )
        )

    algo = config.build()

    # The built-in param storage does not work
    model_param_file_path = osp.join(algo.logdir, 'model_params.json')
    with open(model_param_file_path, 'w') as fp:
        json.dump(RF_MODEL_DEFAULTS, fp, indent=2)

    # algo.restore('/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-215bpvmwd3/checkpoint_000451')
    if args.restore_dir is not None:
        algo.restore(args.restore_dir)
        # /home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-16_02-16-18e5y0w4ou/checkpoint_000801

    train_iter = 1601
    for i in range(train_iter):
        result = algo.train()
        print(pretty_print(result))

        if i % 50 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_model", default='nystrom_q', type=str, help="Choose model from following: nystrom_q, random_feature_q") #
    parser.add_argument("--feature_dim", default=256, type=int)
    parser.add_argument("--env_id", default='Pendubot-v0', type=str)
    parser.add_argument("--algo", default='SAC', type=str)
    parser.add_argument("--reward_exp", default=True, type=bool)
    parser.add_argument("--reward_scale", default=10., type=float)
    parser.add_argument("--noisy", default=False, type=bool)
    parser.add_argument("--noise_scale", default=0., type=float)
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--reward_type", default='lqr', type=str)
    parser.add_argument("--theta_cal", default='sin_cos', type=str)
    parser.add_argument("--comments", default='debug revised custom model', type=str)
    parser.add_argument("--restore_dir",default=None, type=str)
    args = parser.parse_args()
    train_rfsac(args)
    # env = env_creator_pendubot(ENV_CONFIG)
    # print(env.reset())
    # print(env.observation_space)
    # print(env.action_space)
    # action = env.action_space.sample()
    # print(env.step(action))
    # print(env.step(action))
