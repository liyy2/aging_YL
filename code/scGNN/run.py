import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from scGNN.model.aging_model import LightningAgingModel
from scGNN.data.dataset import SingleCellDataModule
import numpy as np
from scGNN.utils.lightening_utils import EpochTestCallback, calculate_stats, EMACallback, ConfigureSchdulerCallback
from scGNN.utils.argparse_utils import convert_argpasere_to_dict


def seed_everything(seed):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Aging Model Training')
    parser.add_argument('--hidden', type=int, default=256, help='hidden layer size')
    parser.add_argument('--num_transformer_layer', type=int, default=6, help='number of transformer layers')
    parser.add_argument('--num_cell_embed_transformer_layer', type=int, default=6, help='number of cell embed transformer layers')
    parser.add_argument('--cat_list', nargs='+', default=[12, 2, 10, 6], help='list of categories')
    parser.add_argument('--max_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=200, help='patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--preprocess', action='store_true', default=False, help='preprocess data')
    parser.add_argument('--process_only', action='store_true', default=False, help='process data')
    parser.add_argument('--num_genes', default=500, type=int, help='number of genes')
    parser.add_argument('--ema_decay', default=0.999, type=float, help='ema decay')
    parser.add_argument('--node_sampling_ratio', default=0.5, type=float, help='node sampling ratio')
    parser.add_argument('--max_node_num', default=2000, type=int, help='max node num')
    parser.add_argument('--min_node_num', default=1000, type=int, help='min node num')
    parser.add_argument('--test_interval', default=10, type=int, help='test interval')
    parser.add_argument('--linear_attn', action='store_true', default=True, help='use linear attention')
    parser.add_argument('--bin_expression', default=10, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Define model
    config = convert_argpasere_to_dict(args)
    seed_everything(args.seed)

    # Define data module
    # Create a DataModule using SingleCellDataset
    # data_module = YourDataModule(dataset, ...)
    data_module = SingleCellDataModule('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices', 
                                       batch_size=1, config=config)
    if args.preprocess:
        data_module.prepare_data()
        if args.process_only:
            return
    # config = calculate_stats(data_module.train_dataset, config)
    config['mean'] = 50.48716354370117
    config['std'] = 25.899734497070312


    model = LightningAgingModel(args.num_genes, args.hidden, args.num_transformer_layer, 
                                args.cat_list, lr = args.lr, config=config)
    

    # Define logger and callbacks
    wandb_logger = WandbLogger(project='AgingModel', log_model=True, config=config)
    early_stopping = EarlyStopping('val_loss', patience=args.patience)
    # epoch_test_callback = EpochTestCallback(test_interval=args.test_interval, datamodule=data_module)
    ema_callback = EMACallback(decay=args.ema_decay, use_ema_weights=True)
    scheduler_config_callback = ConfigureSchdulerCallback(data_module.train_dataloader().__len__())
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[scheduler_config_callback, early_stopping, ema_callback],
        logger=[wandb_logger],
        strategy='ddp_find_unused_parameters_true',
        precision="bf16",
        accumulate_grad_batches=args.batch_size,
        gradient_clip_val=1.0,
        # accelerator='cpu'
        # gpus=-1  # use all available GPUs
    )

    # Train model
    trainer.fit(model, datamodule=data_module)

    # Test
    trainer.test(ckpt_path='best', datamodule=data_module)

if __name__ == '__main__':
    main()

