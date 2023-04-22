import numpy as np

from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.offline import JsonReader


import argparse
from gymnasium.spaces import Box, Discrete
import numpy as np

from ray.rllib.examples.models.custom_model_api import (
    DuelingQModel,
    TorchDuelingQModel,
    ContActionQModel,
    TorchContActionQModel,
)
from ray.rllib.models.catalog import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class TorchCustomLossModel(TorchModelV2, nn.Module):
    """PyTorch version of the CustomLossModel above."""

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, input_files
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.input_files = input_files
        # Create a new input reader per worker.
        self.reader = JsonReader(self.input_files)
        self.fcnet = TorchFC(
            self.obs_space, self.action_space, num_outputs, model_config, name="fcnet"
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Delegate to our FCNet.
        return self.fcnet(input_dict, state, seq_lens)

    @override(ModelV2)
    def value_function(self):
        # Delegate to our FCNet.
        return self.fcnet.value_function()

    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """Calculates a custom loss on top of the given policy_loss(es).

        Args:
            policy_loss (List[TensorType]): The list of already calculated
                policy losses (as many as there are optimizers).
            loss_inputs: Struct of np.ndarrays holding the
                entire train batch.

        Returns:
            List[TensorType]: The altered list of policy losses. In case the
                custom loss should have its own optimizer, make sure the
                returned list is one larger than the incoming policy_loss list.
                In case you simply want to mix in the custom loss into the
                already calculated policy losses, return a list of altered
                policy losses (as done in this example below).
        """
        # Get the next batch from our input files.
        batch = self.reader.next()

        # Define a secondary loss by building a graph copy with weight sharing.
        obs = restore_original_dimensions(
            torch.from_numpy(batch["obs"]).float().to(policy_loss[0].device),
            self.obs_space,
            tensorlib="torch",
        )
        logits, _ = self.forward({"obs": obs}, [], None)

        # You can also add self-supervised losses easily by referencing tensors
        # created during _build_layers_v2(). For example, an autoencoder-style
        # loss can be added as follows:
        # ae_loss = squared_diff(
        #     loss_inputs["obs"], Decoder(self.fcnet.last_layer))
        print("FYI: You can also use these tensors: {}, ".format(loss_inputs))

        # Compute the IL loss.
        action_dist = TorchCategorical(logits, self.model_config)
        imitation_loss = torch.mean(
            -action_dist.logp(
                torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
            )
        )
        self.imitation_loss_metric = imitation_loss.item()
        self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

        # Add the imitation loss to each already calculated policy loss term.
        # Alternatively (if custom loss has its own optimizer):
        # return policy_loss + [10 * self.imitation_loss]
        return [loss_ + 10 * imitation_loss for loss_ in policy_loss]

    def metrics(self):
        return {
            "policy_loss": self.policy_loss_metric,
            "imitation_loss": self.imitation_loss_metric,
        }

