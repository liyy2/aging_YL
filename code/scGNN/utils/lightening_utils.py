from pytorch_lightning import Callback
import numpy as np
import logging
from tqdm import tqdm
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from timm.utils.model import get_state_dict, unwrap_model
from timm.utils.model_ema import ModelEmaV2
from torch.optim.lr_scheduler import OneCycleLR 

class ConfigureSchdulerCallback(Callback):
    def __init__(self, steps_per_epoch=None):
        self.steps_per_epoch = steps_per_epoch
    
    def on_train_init(self, trainer, pl_module):

        optimizer = trainer.optimizers[0]  # Assuming you have only one optimizer

        scheduler = OneCycleLR(
            optimizer,
            max_lr=pl_module.lr,
            epochs=trainer.max_epochs,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        # Resetting any existing schedulers if needed
        pl_module.lr_schedulers = []

        # Adding the new scheduler
        pl_module.lr_schedulers.append({'scheduler': scheduler, 'interval': 'step', 'name': 'learning_rate'})

class EpochTestCallback(Callback):
    def __init__(self, test_interval=50, datamodule=None):
        self.test_interval = test_interval
        self.datamodule = datamodule
    def on_validation_end(self, trainer, pl_module, *args, **kwargs):
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.test_interval == 0:  # +1 because epoch is zero-indexed
            trainer.test(ckpt_path='best', datamodule=self.datamodule)

class EMACallback(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    If `use_ema_weights`, then the ema parameters of the network is set after training end.
    """

    def __init__(self, decay=0.9999, use_ema_weights: bool = True):
        self.decay = decay
        self.ema = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer, pl_module):
        "Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"
        self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)

    def on_train_batch_end(
        self, trainer, pl_module, *args, **kwargs
    ):
        "Update the stored parameters using a moving average"
        # Update currently maintained parameters.
        self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module, *args, **kwargs):
        "do validation using the stored parameters"
        # save original parameters before replacing with EMA version
        self.store(pl_module.parameters())

        # update the LightningModule with the EMA weights
        # ~ Copy EMA parameters to LightningModule
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module, *args, **kwargs):
        "Restore original parameters to resume training later"
        self.restore(pl_module.parameters())

    def on_train_end(self, trainer, pl_module, *args, **kwargs):
        # update the LightningModule with the EMA weights
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
            print("Model weights replaced with the EMA version.")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            return {"state_dict_ema": get_state_dict(self.ema, unwrap_model)}

    def on_load_checkpoint(self, callback_state):
        if self.ema is not None:
            self.ema.module.load_state_dict(callback_state["state_dict_ema"])

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)


def calculate_stats(train_dataset, config):
    all_labels = []
    for data in tqdm(train_dataset):
        all_labels.append(data.y)
    all_labels = np.array(all_labels)
    mean = np.mean(all_labels)
    std = np.std(all_labels)
    logging.info(f'The mean of the age is {mean}, the std of the age is {std}')
    config['mean'] = mean
    config['std'] = std
    return config