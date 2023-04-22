from ray.rllib.algorithms.sac import SACConfig, sac
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from sac_torch_random_feature_model import SACTorchRFModel
from ray.rllib.utils.typing import ModelConfigDict

ModelCatalog.register_custom_model("sac_rf_model", SACTorchRFModel)

RF_MODEL_DEFAULTS: ModelConfigDict = {'random_feature_dim': 256,
                                      'sigma': 0,
                                      'learn_rf': False,
                                      'dynamics_type': 'pendulum',
                                      'dynamics_parameters': {}}

RF_MODEL_DEFAULTS.update(MODEL_DEFAULTS)

algo = sac.SAC(env="Pendulum-v1", config={
    "framework": "torch",
    "q_model_config": RF_MODEL_DEFAULTS
    })

for i in range(10):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
