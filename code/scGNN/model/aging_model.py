from typing import Any, List, Optional, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, LRSchedulerPLType, LRSchedulerTypeUnion
from scGNN.model.cell_embed_transformer import CellEmbedTransformer, CellEmbedTransformerLinearAttention
import torch
from scGNN.model.covariates_embed_net import CovEmbedNet
from scGNN.model.output_net import OutputNet
from scGNN.model.graph_transformer import GraphTransformer
from scGNN.model.loss import *
import pytorch_lightning as pl
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import LearningRateMonitor
from timm.scheduler import create_scheduler_v2


class LightningAgingModel(pl.LightningModule):
    def __init__(self, in_channels, hidden, num_transformer_layer, cat_list, max_cell_type = 30, lr = 1e-3, config = None, **kwargs):
        super().__init__()
        self.hidden = hidden
        self.in_channels = in_channels
        self.cov_embed_net = CovEmbedNet(cat_list, hidden)
        self.output_net = OutputNet(hidden)
        self.graph_transformer = GraphTransformer(hidden, num_transformer_layer)
        self.cell_type_embedding = torch.nn.Embedding(max_cell_type, hidden)
        if config['linear_attn']:
            self.cell_embed_transformer = CellEmbedTransformerLinearAttention(hidden, heads=8, num_layers=config['num_cell_embed_transformer_layer'], config = config)
        else:
            self.cell_embed_transformer = CellEmbedTransformer(hidden, heads=8, num_layers=config['num_cell_embed_transformer_layer'])
        self.num_transformer_layer = num_transformer_layer
        self.loss_fn = Huber(mean_train=config['mean'], std_train=config['std'])
        self.lr = lr
        self.validation_step_output = []
        self.config = config
        self.prepare_data_per_node = config['preprocess']


    def forward(self, data):
        x, edge_index, batch, covariates = data.x, data.edge_index, data.batch, data.covariates
        x = self.cell_embed_transformer(x, data.gene_type, batch)
        x = x + self.cell_type_embedding(data.cell_type)
        covariates_embedding = self.cov_embed_net(covariates)
        x = self.graph_transformer(x, edge_index)
        age_pred = self.output_net(x, edge_index, batch, covariates_embedding)
        return age_pred

    def training_step(self, batch, batch_idx):
        # Forward pass
        age_pred = self.forward(batch)
        loss = self.loss_fn(age_pred, batch.y)
        # Log metrics
        # log lr
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, batch_size=batch.y.shape[0])

        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        age_pred = self.forward(batch)
        loss = self.loss_fn(age_pred, batch.y)
        self.validation_step_output.append({'val_loss': loss, 'preds': age_pred, 'targets': batch.y})
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch.y.shape[0])
        return {'val_loss': loss, 'preds': age_pred, 'targets': batch.y}

    def on_validation_epoch_end(self):
        self.compute_metrics(self.validation_step_output, 'val')
        self.validation_step_output = []

    def test_step(self, batch, batch_idx):
        age_pred = self.forward(batch)
        loss = self.loss_fn(age_pred, batch.y)
        self.validation_step_output.append({'test_loss': loss, 'preds': age_pred, 'targets': batch.y})
        return {'test_loss': loss, 'preds': age_pred, 'targets': batch.y}

    def on_test_epoch_end(self):
        self.compute_metrics(self.validation_step_output, 'test')
        self.validation_step_output = []

    def compute_metrics(self, outputs, prefix):
        all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
        all_targets = torch.cat([x['targets'] for x in outputs], dim=0)
        
        # Gather all predictions and targets across GPUs
        all_preds = self.all_gather(all_preds)
        all_targets = self.all_gather(all_targets)

        if self.global_rank == 0:
            age_pred_cpu = all_preds.to(dtype=torch.float32).detach().cpu().numpy()
            all_targets_cpu = all_targets.to(dtype=torch.float32).detach().cpu().numpy()
            
            # flatten
            age_pred_cpu = np.concatenate(age_pred_cpu).squeeze() if len(age_pred_cpu.shape) >= 2 else age_pred_cpu
            all_targets_cpu = np.concatenate(all_targets_cpu).squeeze() if len(all_targets_cpu.shape) >= 2 else all_targets_cpu
            # Compute RMSE
            rmse = np.sqrt(np.mean((age_pred_cpu * self.config['std'] + self.config['mean'] - all_targets_cpu) ** 2))
            self.log(f'{prefix}_rmse', rmse, on_epoch=True)

            # Compute MAE
            mae = np.mean(np.abs(age_pred_cpu * self.config['std'] + self.config['mean'] - all_targets_cpu))
            self.log(f'{prefix}_mae', mae, on_epoch=True)

            # Compute Pearson correlation
            age_pred_cpu = age_pred_cpu.squeeze() if len(age_pred_cpu.shape) == 2 else age_pred_cpu
            all_targets_cpu = all_targets_cpu.squeeze() if len(all_targets_cpu.shape) == 2 else all_targets_cpu
            pearson_corr, _ = pearsonr(age_pred_cpu, all_targets_cpu)
            self.log(f'{prefix}_pearson', pearson_corr, on_epoch=True)

            # Compute Spearman correlation
            spearman_corr, _ = spearmanr(age_pred_cpu, all_targets_cpu)
            self.log(f'{prefix}_spearman', spearman_corr, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler_config = {
        # REQUIRED: The scheduler instance
        "scheduler": create_scheduler_v2(
            optimizer,
            num_epochs= self.config['max_epochs'],
            decay_epochs= self.config['max_epochs'] - self.config['lr_warmup_epochs'],
            warmup_epochs= self.config['lr_warmup_epochs'],
            min_lr=1e-6,
            patience_epochs=30,)[0],
        # The unit of the scheduler's step size, could also be 'step'.
        # 'epoch' updates the scheduler on epoch end whereas 'step'
        # updates it after a optimizer update.
        "interval": "epoch",
        # How many epochs/steps should pass between calls to
        # `scheduler.step()`. 1 corresponds to updating the learning
        # rate after every epoch/step.
        "frequency": 1,
        # Metric to to monitor for schedulers like `ReduceLROnPlateau`
        "monitor": "val_loss",
        # If set to `True`, will enforce that the value specified 'monitor'
        # is available when the scheduler is updated, thus stopping
        # training if not found. If set to `False`, it will only produce a warning
        "strict": True,
        # If using the `LearningRateMonitor` callback to monitor the
        # learning rate progress, this keyword can be used to specify
        # a custom logged name
        "name": 'lr',
        }
        return [optimizer], [lr_scheduler_config]

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any | None) -> None:
        scheduler.step(self.current_epoch)






