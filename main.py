from ray.rllib.algorithms.sac import SACConfig, sac
from ray.tune.logger import pretty_print


algo = (
    SACConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="CartPole-v1")
    .framework('torch')
    .training(twin_q=False)
    .build()
)

# algo = sac.SAC(env="CartPole-v1", config={
#     ""
#     "framework": "torch",
#     "custom_model": "sac_rf_model",
#     "custom_model_config":{
#         "random_feature_dim": 256,
#
#     }
# })

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")