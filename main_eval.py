import gymnasium
import ray
from ray.rllib.algorithms.sac import SACConfig, sac, RFSACConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import register_env

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
from main import TransformTriangleObservationWrapper, TransformDoubleTriangleObservationWrapper
import torch
import time

from copy import deepcopy

ModelCatalog.register_custom_model("sac_rf_model", SACTorchRFModel)

RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 8192,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'Quadrotor2D', # Pendulum, Quadrotor2D
                                      'dynamics_parameters': {
                                                              # 'stabilizing_target': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                                              # 'reward_exponential': REWARD_EXP,
                                                              # 'reward_scale': REWARD_SCALE,
                                                              }}

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)

ENV_CONFIG = {'sin_input': True,
              'reward_exponential': False,
              'reward_scale': 10.,
              'reward_type' : 'energy',
              'theta_cal': 'sin_cos',
              'render': False
              }

RF_MODEL_DEFAULTS.update(ENV_CONFIG)
RF_MODEL_DEFAULTS.get('dynamics_parameters').update(ENV_CONFIG)

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
    env = gymnasium.make('CartPoleContinuous-v0', render_mode='human') #
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
                         theta_cal=env_config.get('theta_cal'),
                         render_mode='human' if env_config.get('render') else None
                         ) #
    env = TransformReward(env, lambda r: reward_scale_pendubot * r)
    if env_config.get('reward_exponential'):
        env = TransformReward(env, lambda r: np.exp(r))
    if env_config.get('sin_input'):
        return TransformDoubleTriangleObservationWrapper(env)
    else:
        return env

def train_rfsac(args):
    ray.init(local_mode=True)
    RF_MODEL_DEFAULTS.update({'random_feature_dim': args.random_feature_dim})
    RF_MODEL_DEFAULTS.update({'dynamics_type' : args.env_id.split('-')[0]})
    ENV_CONFIG.update({
                        'reward_exponential':args.reward_exp,
                        'reward_type': args.reward_type,
                        'reward_scale': args.reward_scale,
                        'theta_cal': args.theta_cal
                      })
    RF_MODEL_DEFAULTS['dynamics_parameters'].update(ENV_CONFIG)
    RF_MODEL_DEFAULTS.update(ENV_CONFIG) # todo:not update twice
    RF_MODEL_DEFAULTS.update({'comments': args.comments})

    register_env('Quadrotor2D-v1', env_creator)
    register_env('CartPoleContinuous-v0', env_creator_cartpole)
    register_env('Pendubot-v0', env_creator_pendubot)

    if args.algo == 'RFSAC':
        config = RFSACConfig().environment(env=args.env_id, env_config=ENV_CONFIG)\
            .framework("torch") .training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=1)

    elif args.algo == 'SAC':
        config = SACConfig().environment(env=args.env_id, env_config=ENV_CONFIG)\
            .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=1)

    if args.eval:
        # eval_config = RFSACConfig.overrides(env_config=ENV_CONFIG)
        config = config.evaluation(
            # evaluation_parallel_to_training=True,
            evaluation_interval=10,
            evaluation_duration=10,
            evaluation_num_workers=1,
            evaluation_config=RFSACConfig.overrides(render_env=False,
                                                    env_config={
                                                        'sin_input': True,
                                                        'reward_exponential': True,
                                                        'reward_scale': 10.,
                                                        'reward_type': 'energy',
                                                    }
                                                    )
        )

    algo = config.build()

    # The built-in param storage does not work
    model_param_file_path = osp.join(algo.logdir, 'model_params.json')
    with open(model_param_file_path, 'w') as fp:
        json.dump(RF_MODEL_DEFAULTS, fp, indent=2)

    # algo.restore('/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-215bpvmwd3/checkpoint_000451')
    # algo.restore('/home/mht/ray_results/SAC_CartPoleContinuous-v0_2023-05-29_06-51-17i_cw3m6f/checkpoint_000451')
    algo.restore('/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-16_02-49-15fflqt3si/checkpoint_001602')
    # algo.restore('/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-19_12-27-51d40yaq_b/checkpoint_001501')
    policy = algo.workers.local_worker().policy_map['default_policy']
    returns = []
    eval_env_config = {'sin_input': True,
                      'reward_exponential': False,
                      'reward_scale': 1.,
                      'reward_type' : 'energy',
                      'theta_cal': 'sin_cos',
                      'noisy': True,
                      'noise_scale': 0.5,
                       'render': False
                      }
    env = env_creator_pendubot(eval_env_config)
    for eval_epis in range(10):
        ret = 0.
        obs, _ = env.reset(seed=eval_epis)
        for _ in range(200):
            input_ = np.array([obs])
            # Note that for PyTorch, you will have to provide torch tensors here.
            # if args.framework == "torch":
            input_ = torch.from_numpy(input_)
            input_dict = SampleBatch(obs=input_, _is_training=False)
            action, _, _ = policy.compute_actions_from_input_dict(input_dict=input_dict, explore=False)
            obs, reward, terminated, done, info = env.step(action[0])
            ret += reward
        print(ret)
        returns.append(ret)
    print("{:.3f} $\pm$ {:.3f}".format(np.mean(returns), np.std(returns)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_feature_dim", default=32768, type=int)
    parser.add_argument("--env_id", default='Pendubot-v0', type=str)
    parser.add_argument("--algo", default='RFSAC', type=str)
    parser.add_argument("--reward_exp", default=True, type=str)
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--reward_scale", default=10., type=float)
    parser.add_argument("--reward_type", default='lqr', type=str)
    parser.add_argument("--theta_cal", default='arctan', type=str)
    parser.add_argument("--comments", default='train with lqr and eval with energy, both exp', type=str)
    args = parser.parse_args()
    train_rfsac(args)
    # env = env_creator_cartpole(ENV_CONFIG)
    # print(env.reset())
    # print(env.observation_space)
    # print(env.action_space)
    # action = env.action_space.sample()
    # print(env.step(action))
    # print(env.step(action))
