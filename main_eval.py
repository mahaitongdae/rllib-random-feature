import os

import gymnasium
import ray
from ray.rllib.algorithms.sac import SACConfig, sac, RFSACConfig
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.policy.sample_batch import SampleBatch
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
                                                              },
                                      'kernel_represetation': 'random_feature',
                                      'seed': 0,
                                      }

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)

ENV_CONFIG = {'sin_input': True,
              'reward_exponential': False,
              'reward_scale': 10.,
              'reward_type' : 'energy',
              'theta_cal': 'sin_cos',
              'render': False,
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
    env = gymnasium.make('CartPoleContinuous-v0',
                         render_mode='human' if env_config.get('render') else None) #
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
    exp_path = os.path.dirname(args.restore_path)
    json_path = os.path.join(exp_path, 'model_params.json')
    algo = exp_path.split('/')[-1].split('_')[0]
    env_id = exp_path.split('/')[-1].split('_')[1]
    with open(json_path) as json_file:
        model_params = json.load(json_file)
    RF_MODEL_DEFAULTS.update(model_params)
    for key, value in model_params.items():
        if key in ENV_CONFIG.keys():
            ENV_CONFIG.update({key: value})

    register_env('Quadrotor2D-v1', env_creator)
    register_env('CartPoleContinuous-v0', env_creator_cartpole)
    register_env('Pendubot-v0', env_creator_pendubot)
    register_env('Pendulum-v1', env_creator_pendulum)

    if algo == 'RFSAC':
        config = RFSACConfig().environment(env=env_id, env_config=ENV_CONFIG)\
            .framework("torch") .training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=1)

    elif algo == 'SAC':
        config = SACConfig().environment(env=env_id, env_config=ENV_CONFIG)\
            .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=1)

    algo = config.build()

    # algo.restore('/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-215bpvmwd3/checkpoint_000451')
    # algo.restore('/home/mht/ray_results/SAC_CartPoleContinuous-v0_2023-05-29_06-51-17i_cw3m6f/checkpoint_000451')

    # algo.restore(
    #     '/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-16_02-49-15fflqt3si/checkpoint_001602')  # energy noisy
    # algo.restore(
    #     '/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-19_12-27-51d40yaq_b/checkpoint_001501') # lqr non noisy
    # algo.restore(
    #     '/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-16_00-44-0154jj9f05/checkpoint_001502')  # lqr noisy
    eval_env_config = {'sin_input': True,
                       'reward_exponential': False,
                       'reward_scale': 1.,
                       'reward_type': 'energy',
                       'theta_cal': 'sin_cos',
                       'noisy': True,
                       'noise_scale': 0.5,
                       'render': True
                       }

    n = 4
    m = 1


    def run_simulation_for_perf(path):
        algo.restore(path)
        policy = algo.workers.local_worker().policy_map['default_policy']
        returns = []

        env_creator_func = _global_registry.get(ENV_CREATOR,
                                                env_id)  # from algorithm.py line 2212, Algorithm.__init__()
        env = env_creator_func(eval_env_config)
        for eval_epis in range(50):
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


    def run_simulation_to_plot(path, N = 200, ):
        s = np.zeros((N + 1, n))
        u = np.zeros((N, m))
        algo.restore(path)
        policy = algo.workers.local_worker().policy_map['default_policy']
        ret = 0
        env = env_creator_pendubot(eval_env_config)
        obs, _ = env.reset(seed=0)
        s[0] = env.get_state()
        for i in range(N):
            input_ = np.array([obs])
            input_ = torch.from_numpy(input_)
            input_dict = SampleBatch(obs=input_, _is_training=False)
            action, _, _ = policy.compute_actions_from_input_dict(input_dict=input_dict, explore=False)
            obs, reward, terminated, done, info = env.step(action[0])
            s[i + 1] = env.get_state()
            u[i] = action[0]
            ret += reward
        print(ret)

        return s, u

    def run_simulation_to_plot_energy(N = 200, ):
        s = np.zeros((N + 1, n))
        u = np.zeros((N, m))
        from envs.pendubot import energy_based_controller
        ret = 0
        env = env_creator_pendubot(eval_env_config)
        obs, _ = env.reset(seed=0)
        s[0] = env.get_state()
        for i in range(N):
            input_ = np.array([obs])
            input_ = torch.from_numpy(input_)
            input_dict = SampleBatch(obs=input_, _is_training=False)
            action = energy_based_controller(env) / env.force_mag
            obs, reward, terminated, done, info = env.step(action)
            s[i + 1] = env.get_state()
            u[i] = action
            ret += reward
        print(ret)

        return s, u

    def plot_data():
        t = np.arange(0., 10.05, 0.05)
        s_energy, u_energy = run_simulation_to_plot('/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-16_02-49-15fflqt3si/checkpoint_001602')
        s_lqr, u_lqr = run_simulation_to_plot('/home/mht/ray_results/RFSAC_Pendubot-v0_2023-06-16_00-44-0154jj9f05/checkpoint_001502')
        s_non, u_non = run_simulation_to_plot_energy()

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, n + m, dpi=150, figsize=(15, 3))
        plt.subplots_adjust(wspace=0.45)
        labels_s = (r'$q_1(t)$', r'$q_2(t)$', r'$\dot{q}_1(t)$', r'$\dot{q}_2(t)$')
        labels_u = (r'$u(t)$',)
        for i in range(n):
            axes[i].plot(t, s_energy[:, i], label='learning energy')
            axes[i].plot(t, s_lqr[:, i], label='learning quad')
            axes[i].plot(t, s_non[:, i], label='energy-based')
            axes[i].set_xlabel(r'$t$')
            axes[i].set_ylabel(labels_s[i])

        for i in range(m):
            axes[n + i].plot(t[:-1], u_energy[:, i], label='learning energy')
            axes[n + i].plot(t[:-1], u_lqr[:, i], label='learning quad')
            axes[n + i].plot(t[:-1], u_non[:, i], label='energy-based')
            axes[n + i].set_xlabel(r'$t$')
            axes[n + i].set_ylabel(labels_u[i])
            if i == 0:
                axes[i].legend(loc='best')
        # if closed_loop:
        plt.tight_layout()
        plt.savefig('traj.png') # , bbox_inches='tight'
        # else:
        #     plt.savefig('cartpole_swingup_ol.png', bbox_inches='tight')
        plt.show()

    # plot_data() # for plot comparison data in pendubot

    run_simulation_for_perf(args.restore_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--random_feature_dim", default=32768, type=int)
    # parser.add_argument("--env_id", default='Pendubot-v0', type=str)
    # parser.add_argument("--algo", default='RFSAC', type=str)
    # parser.add_argument("--reward_exp", default=True, type=str)
    # parser.add_argument("--eval", default=False, type=bool)
    # parser.add_argument("--reward_scale", default=10., type=float)
    # parser.add_argument("--reward_type", default='lqr', type=str)
    # parser.add_argument("--theta_cal", default='arctan', type=str)
    parser.add_argument("--restore_path", default='/home/mht/ray_results/RFSAC_Pendulum-v1_2023-07-18_11-33-35dovuxm6k/checkpoint_001001', type=str)
    # parser.add_argument("--comments", default='train with lqr and eval with energy, both exp', type=str)
    args = parser.parse_args()
    train_rfsac(args)
    # env = env_creator_cartpole(ENV_CONFIG)
    # print(env.reset())
    # print(env.observation_space)
    # print(env.action_space)
    # action = env.action_space.sample()
    # print(env.step(action))
    # print(env.step(action))
