import gymnasium
import ray
from ray.rllib.algorithms.sac import SACConfig, RFSACConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
# from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env, ENV_CREATOR, _global_registry
import ray

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

from copy import deepcopy

# ModelCatalog.register_custom_model("sac_rf_model", SACTorchRFModel)

RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 8192,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'Quadrotor2D', # Pendulum, Quadrotor2D
                                      'dynamics_parameters': {
                                                              'stabilizing_target': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                                              # 'reward_exponential': REWARD_EXP,
                                                              # 'reward_scale': REWARD_SCALE,
                                                              },
                                      'restore_dir': None}

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)

ENV_CONFIG = {'sin_input': True,
              'reward_exponential': False,
              'reward_scale': 10.,
              'reward_type' : 'energy',
              'theta_cal': 'sin_cos',
              'noisy': False,
              'noise_scale': 0.
              }

RF_MODEL_DEFAULTS.update(ENV_CONFIG)
RF_MODEL_DEFAULTS.get('dynamics_parameters').update(ENV_CONFIG)

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

    def observation(self, observation):
        '''
        transfer observations. We assume that the last two observations is the angle and angular velocity.
        '''
        theta = observation[-2]
        sin_cos_theta = np.array([np.cos(theta), np.sin(theta)])
        theta_dot = observation[-1:]
        return np.hstack([observation[:-2], sin_cos_theta, theta_dot])

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

def env_creator_pendulum(env_config):
    env = gymnasium.make('Pendulum-v1')
    if env_config.get('reward_exponential'):
        env = TransformReward(env, lambda r: np.exp(env_config.get('reward_scale') * r))
    else:
        env = TransformReward(env, lambda r: env_config.get('reward_scale') * r)
    return env

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
    ray.init(local_mode=True) # local_mode=True
    # RF_MODEL_DEFAULTS.update({'random_feature_dim': args.random_feature_dim})
    RF_MODEL_DEFAULTS.update({'dynamics_type' : args.env_id.split('-')[0]})
    ENV_CONFIG.update({
                        'reward_exponential':args.reward_exponential,
                        'reward_type': args.reward_type,
                        'reward_scale': args.reward_scale,
                        'theta_cal': args.theta_cal
                      })
    RF_MODEL_DEFAULTS['dynamics_parameters'].update(ENV_CONFIG)
    RF_MODEL_DEFAULTS.update(ENV_CONFIG) # todo:not update twice
    RF_MODEL_DEFAULTS.update({'comments': args.comments,
                              'kernel_representation': args.kernel_representation,
                              'nystrom_sample_dim': args.nystrom_sample_dim,
                              'seed':args.seed})

    RF_MODEL_DEFAULTS.update(vars(args))

    register_env('Quadrotor2D-v1', env_creator)
    register_env('CartPoleContinuous-v0', env_creator_cartpole)
    register_env('Pendubot-v0', env_creator_pendubot)
    register_env('Pendulum-v1', env_creator_pendulum)

    env_creator_func = _global_registry.get(ENV_CREATOR,
                                            args.env_id)  # from algorithm.py line 2212, Algorithm.__init__()
    env = env_creator_func(ENV_CONFIG)
    RF_MODEL_DEFAULTS.update({
        'obs_space_high': np.clip(env.observation_space.high, -10., 10.).tolist(),
        'obs_space_low': np.clip(env.observation_space.low, -10., 10.).tolist(), # in case of inf observation space
        'obs_space_dim': env.observation_space.shape,
    })
    del env

    if args.algo == 'RFSAC':
        config = RFSACConfig().environment(env=args.env_id, env_config=ENV_CONFIG)\
            .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=1) #

    elif args.algo == 'SAC':
        config = SACConfig().environment(env=args.env_id, env_config=ENV_CONFIG)\
            .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS,
                                         ).rollouts(num_rollout_workers=12)

    # if args.eval:
    eval_env_config = copy.deepcopy(ENV_CONFIG)
    eval_env_config.update({
                            'sin_input': True,
                            'reward_exponential': False,
                            'reward_scale': 1.,
                            'reward_type': 'energy',
                            })
    config = config.evaluation(
        evaluation_parallel_to_training=True,
        evaluation_interval=5,
        evaluation_duration=50,
        evaluation_num_workers=2,
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

    train_iter = args.train_iter if args.train_iter else 1001
    for i in range(train_iter):
        result = algo.train()
        print(pretty_print(result))

        if i % 500 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_feature_dim", default=256, type=int)
    parser.add_argument("--env_id", default='Pendulum-v1', type=str)
    parser.add_argument("--algo", default='RFSAC', type=str)
    parser.add_argument("--reward_exponential", default=False, type=bool)
    parser.add_argument("--reward_scale", default=0.2, type=float)
    parser.add_argument("--noisy", default=False, type=bool)
    parser.add_argument("--noise_scale", default=0., type=float)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--reward_type", default='energy', type=str)
    parser.add_argument("--theta_cal", default='sin_cos', type=str)
    parser.add_argument("--comments", default='test using parameter to store samples', type=str)
    parser.add_argument("--restore_dir",default=None, type=str)
    parser.add_argument("--kernel_representation", default='nystrom', type=str)
    parser.add_argument("--nystrom_sample_dim",
                        default=512,
                        type=int,
                        help='if use nystrom, sampling on the nystrom_sample_dim. Should be larger than random_feature_dim')
    parser.add_argument("--train_iter", default=1001, type=int)
    args = parser.parse_args()
    train_rfsac(args)
    # env_creator_func = _global_registry.get(ENV_CREATOR,
    #                                         'Pendulum-v1')
    # env = env_creator_pendubot(ENV_CONFIG)
    # env = gymnasium.make('Pendulum-v1')
    # print(env.reset())
    # print(env.observation_space)
    # print(env.action_space)
    # action = env.action_space.sample()
    # print(env.step(action))
    # print(env.step(action))
