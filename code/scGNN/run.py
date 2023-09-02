import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from scGNN.model.aging_model import LightningAgingModel
from scGNN.data.dataset import SingleCellDataModule
import numpy as np
from scGNN.utils.lightening_utils import EpochTestCallback, calculate_stats
from scGNN.utils.argparse_utils import convert_argpasere_to_dict





def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Aging Model Training')
    parser.add_argument('--hidden', type=int, default=64, help='hidden layer size')
    parser.add_argument('--num_transformer_layer', type=int, default=3, help='number of transformer layers')
    parser.add_argument('--cat_list', nargs='+', default=[12, 2, 10, 6], help='list of categories')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Define model
    config = convert_argpasere_to_dict(args)


    # Define data module
    # Create a DataModule using SingleCellDataset
    # data_module = YourDataModule(dataset, ...)
    config = calculate_stats(data_module.train_dataset, config)


    data_module = SingleCellDataModule('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices', batch_size=args.batch_size)
    model = LightningAgingModel(args.hidden, args.num_transformer_layer, args.cat_list, args.lr)

    # Define logger and callbacks
    wandb_logger = WandbLogger(project='AgingModel', log_model=True)
    early_stopping = EarlyStopping('val_loss', patience=args.patience)
    epoch_test_callback = EpochTestCallback()

    # Trainer
    trainer = pl.Trainer.from_argparse_args(args,
                                            max_epochs=args.max_epochs,
                                            callbacks=[early_stopping, epoch_test_callback],
                                            logger=[wandb_logger],
                                            accelerator='ddp',
                                            gpus=-1)  # use all available GPUs

    # Train model
    trainer.fit(model, datamodule=data_module)

