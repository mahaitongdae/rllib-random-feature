from ray import tune

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

REWARD_SCALE = 0.1

RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 5000,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'quadrotor_2d', # pendulum, quadrotor_2d
                                      'dynamics_parameters': {'stabilizing_target': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                                                              'reward_exponential': False,
                                                              'reward_scale': REWARD_SCALE,
                                                              }}

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)

def env_creator(env_config):
    CONFIG_FACTORY = ConfigFactory()
    CONFIG_FACTORY.parser.set_defaults(overrides=['./quad_2d_env_config/stabilization.yaml'])
    config = CONFIG_FACTORY.merge()
    env = make('quadrotor', **config.quadrotor_config)
    return TransformReward(env, lambda r: REWARD_SCALE * r)

def train_rfsac(config):
    RF_MODEL_DEFAULTS.update(config)
    register_env('Quadrotor-v1', env_creator)

    config = RFSACConfig().environment(env='Quadrotor-v1') \
        .framework("torch").training(q_model_config=RF_MODEL_DEFAULTS).rollouts(num_rollout_workers=4)

    algo = config.build()

    # algo.restore('/home/mht/ray_results/SAC_Quadrotor-v1_2023-04-24_00-36-16cjy5vsgs/checkpoint_000496')

    for i in range(10):
        result = algo.train()
        print(pretty_print(result))

        tune.report(episode_reward_mean = result.get('episode_reward_mean'))

        if i % 50 == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")


search_space = {
    "random_feature_dim": tune.grid_search([4096, 8192]),
}

trainable_with_resources = tune.with_resources(train_rfsac, {"cpu": 2})
tuner = tune.Tuner(
    trainable_with_resources,
    param_space=search_space,

)
results = tuner.fit()

dfs = {result.log_dir: result.metrics_dataframe for result in results}
[d.episode_reward_mean.plot() for d in dfs.values()]
