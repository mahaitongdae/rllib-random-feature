import time

import numpy as np
import torch
import gymnasium as gym
import argparse
import os
from ray.rllib.utils.checkpoints import CHECKPOINT_VERSION, get_checkpoint_info
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.sac.sac_torch_policy import RFSACTorchPolicy, build_policy_class
from ray.rllib.algorithms.sac.rfsac import RFSACConfig
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from ray.rllib.utils.typing import ModelConfigDict
from main import env_creator_cartpole


from copy import deepcopy
import time

RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 2048,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'CartPoleContinuous', # Pendulum, Quadrotor2D
                                      'sin_input': False}

def main(checkpoint, max_steps=200, state_dim=None):
    checkpoint_info = get_checkpoint_info(checkpoint)
    checkpoint_data = Algorithm._checkpoint_info_to_algorithm_state(
        checkpoint_info
    )
    env = env_creator_cartpole({})
    policy = RFSACTorchPolicy(env.observation_space, env.action_space, RFSACConfig().update_from_dict({'q_model_config': RF_MODEL_DEFAULTS}))
    policy.set_state(checkpoint_data["worker"]["policy_states"]["default_policy"])

    for i in np.arange(1):
        # init_state = init_states[i]
        # state = env.reset(init_state=init_state) if args.env == 'Pendulum-v1' else env.reset()
        state, _ = env.reset()
        eps_reward = 0
        for t in range(max_steps):
            print("current state", state)
            action = policy.compute_actions(state)
            # action = env.action_space.sample()
            # action = agent.select_action(np.array(state))
            print("current action", action)
            state, reward, _, _, _ = env.step(action)
            env.render()
            eps_reward += reward
        # all_rewards[i] = eps_reward

    # print(f"mean episodic reward over 200 time steps (rf_num = {rf_num}, learn_rf = {learn_rf}): ",
    #       np.mean(all_rewards))

def sample_with_worker():
    import gymnasium as gym
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    from ray.rllib.algorithms.pg.pg_torch_policy import PGTorchPolicy
    from ray.rllib.algorithms.algorithm import AlgorithmConfig
    config = AlgorithmConfig().framework("torch")
    worker = RolloutWorker(env_creator = lambda _: gym.make("CartPole-v1"),default_policy_class = PGTorchPolicy, config = config)
    print(worker.sample())  # doctest: +SKIP


if __name__ == "__main__":
    checkpoint = '/home/mht/ray_results/RFSAC_CartPoleContinuous-v0_2023-05-29_06-37-21lnb8dai3/checkpoint_000151'
    # main(checkpoint)
    sample_with_worker()