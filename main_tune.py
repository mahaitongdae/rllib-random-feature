import gymnasium
import ray
from gymnasium.core import ActType, ObsType
from ray.rllib.algorithms.sac import SACConfig, RFSACConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
# from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict
from ray.tune.registry import register_env, ENV_CREATOR, _global_registry, register_trainable
from ray.rllib.algorithms.sac.rfsac import RFSAC
import ray
from ray import tune, air
from ray.air import session

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
from utils import custom_log_creator

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

class NoisyObservationWrapper(gymnasium.Wrapper):

    def __init__(self, env, noise_scale):
        super().__init__(env)
        # np.random.seed(seed)
        self.noise_scale = noise_scale

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        self.env.state[0] = self.env.state[0] + self.np_random.normal(scale=self.noise_scale) * self.env.dt
        return self._get_obs(), reward, done, terminated, info



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
    if env_config.get('noisy'):
        env = NoisyObservationWrapper(env, noise_scale=env_config.get('noise_scale', 1))
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

def train_rfsac(config):
    # local_mode=True
    # RF_MODEL_DEFAULTS.update({'random_feature_dim': args.random_feature_dim})
    RF_MODEL_DEFAULTS.update({'dynamics_type' : config.get('env_id').split('-')[0]})
    ENV_CONFIG.update({
                        key: config.get(key) for key in ['reward_exponential',
                                                         'reward_type',
                                                         'reward_scale',
                                                         'theta_cal']
                      })
    RF_MODEL_DEFAULTS['dynamics_parameters'].update(ENV_CONFIG)
    RF_MODEL_DEFAULTS.update(ENV_CONFIG) # todo:not update twice
    # RF_MODEL_DEFAULTS.update({'comments': args.comments,
    #                           'kernel_representation': args.kernel_representation,
    #                           'nystrom_sample_dim': args.nystrom_sample_dim,
    #                           'seed':args.seed})

    RF_MODEL_DEFAULTS.update(config)

    register_env('Quadrotor2D-v1', env_creator)
    register_env('CartPoleContinuous-v0', env_creator_cartpole)
    register_env('Pendubot-v0', env_creator_pendubot)
    register_env('Pendulum-v1', env_creator_pendulum)
    register_trainable('RFSAC', RFSAC)

    env_creator_func = _global_registry.get(ENV_CREATOR,
                                            config.get('env_id'))  # from algorithm.py line 2212, Algorithm.__init__()
    env = env_creator_func(ENV_CONFIG)
    RF_MODEL_DEFAULTS.update({
        'obs_space_high': np.clip(env.observation_space.high, -10., 10.).tolist(),
        'obs_space_low': np.clip(env.observation_space.low, -10., 10.).tolist(), # in case of inf observation space
        'obs_space_dim': env.observation_space.shape,
    })
    del env

    focus_param_keys = config.get('focus_param_keys', [])
    custom_str = ''
    for key in focus_param_keys:
        custom_str = custom_str + key
        custom_str = custom_str + config.get(key, '')

    if config.get('algo') == 'RFSAC':
        algo_name = config.get('algo') + '_' \
                    + config.get('kernel_representation') + '_' \
                    + str(config.get('random_feature_dim')) + '_' \
                    + str(config.get('nystrom_sample_dim'))
        custom_path = '/home/mht/ray_results/{}/{}'.format(config.get('env_id'), algo_name
                                                           )
        config = RFSACConfig()\
            .debugging(logger_creator=custom_log_creator(custom_path, custom_str))\
            .resources(num_cpus_per_worker=1,
                                         num_gpus=0,
                                         num_cpus_for_local_worker=1)\
            .environment(env=config.get('env_id'),
                         env_config=ENV_CONFIG,
                         )\
            .framework("torch")\
            .training(q_model_config=RF_MODEL_DEFAULTS,
                      optimization_config={
                         'actor_learning_rate': 3e-4,
                         'critic_learning_rate': 3e-4,
                         'entropy_learning_rate': config.get('energy_lr'),}
                                         )\
            .rollouts(num_rollout_workers=config.get('num_rollout_workers'),
                      create_env_on_local_worker=True)

    elif config.get('algo') == 'SAC':
        custom_path = '/home/mht/ray_results/{}/SAC'.format(config.get('env_id'))
        config = SACConfig().debugging(logger_creator=custom_log_creator(custom_path, custom_str))\
            .environment(env=config.get('env_id'),
                                         env_config=ENV_CONFIG)\
            .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS,
                                         ).rollouts(num_rollout_workers=12)


    eval_env_config = copy.deepcopy(ENV_CONFIG)
    eval_env_config.update({
                            'sin_input': True,
                            'reward_exponential': False,
                            'reward_scale': 1.,
                            'reward_type': 'energy',
                            })
    config = config.evaluation(
        # evaluation_parallel_to_training=True,
        evaluation_interval=25,
        evaluation_duration=50,
        # evaluation_num_workers=1,
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
    if config.get('restore_dir') is not None:
        algo.restore(config.get('restore_dir'))
        # /home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-16_02-16-18e5y0w4ou/checkpoint_000801

    train_iter = config.get('train_dir') if config.get('train_dir') else 1001
    # tune.Tuner('RFSAC',
    #            run_config=air.RunConfig(stop={'training_iteration': 10}),
    #            param_space=config.to_dict()).fit()
    for i in range(train_iter):
        result = algo.train()
        session.report(result)

        if i % 500 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_feature_dim", default=512, type=int)
    parser.add_argument("--env_id", default='Pendulum-v1', type=str)
    parser.add_argument("--algo", default='RFSAC', type=str)
    parser.add_argument("--reward_exponential", default=False, type=bool)
    parser.add_argument("--reward_scale", default=0.2, type=float)
    parser.add_argument("--noisy", default=True, type=bool)
    parser.add_argument("--noise_scale", default=1., type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--energy_lr", default=3e-4, type=float)
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
    parser.add_argument("--train_iter", default=1002, type=int)
    parser.add_argument("--num_rollout_workers", default=7, type=int)
    args = parser.parse_args()
    config = vars(args)
    ray.init(num_cpus=16) # num_cpus=12 # , resources={'custom_resources': 2}
    config.update({'random_feature_dim': tune.grid_search([512, 256])})
    # config.update({'seed': tune.grid_search([1, 2, 3, 4])})
    # config.update({'noisy': tune.grid_search([True])})
    trainable_with_resources = tune.with_resources(train_rfsac,
                                                   resources=
                                                    tune.PlacementGroupFactory(
                                                        [{'CPU': 1.0}] + [{'CPU': 1.0}] * args.num_rollout_workers, strategy="PACK")
                                                   ) # the doc is not very clear about this.
    tuner = tune.Tuner(trainable_with_resources,
                       param_space=config,
                       tune_config=tune.TuneConfig(num_samples=1,),
                       run_config=air.RunConfig(stop={'training_iteration': args.train_iter})
                       )
    results = tuner.fit()
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
