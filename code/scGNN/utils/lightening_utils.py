from pytorch_lightning import Callback
import numpy as np
import logging
from tqdm import tqdm

class EpochTestCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % 50 == 0:  # +1 because epoch is zero-indexed
            trainer.test(ckpt_path='best', datamodule=pl_module)

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