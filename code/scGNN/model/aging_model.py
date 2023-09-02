import torch
from scGNN.model.covariates_embed_net import CovEmbedNet
from scGNN.model.output_net import OutputNet
from scGNN.model.graph_transformer import GraphTransformer
from scGNN.model.loss import *
import pytorch_lightning as pl
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr


class LightningAgingModel(pl.LightningModule):
    def __init__(self, in_channels, hidden, num_transformer_layer, cat_list, max_cell_type = 24, lr = 1e-3, config = None, **kwargs):
        super().__init__()
        self.hidden = hidden
        self.in_channels = in_channels
        self.cov_embed_net = CovEmbedNet(cat_list, hidden)
        self.output_net = OutputNet(hidden)
        self.graph_transformer = GraphTransformer(in_channels, hidden, num_transformer_layer)
        self.cell_type_embedding = torch.nn.Embedding(max_cell_type, hidden)
        self.num_transformer_layer = num_transformer_layer
        self.loss_fn = Huber(mean_train=config['mean'], std_train=config['std'])
        self.lr = lr
        self.val_rmse = []
        self.val_pearson = []
        self.val_mae = []
        self.val_spearman = []
        self.config = config


    def forward(self, data):
        x, edge_index, batch, covariates = data.x, data.edge_index, data.batch, data.covariates
        x = x + self.cell_type_embedding(data.cell_type)
        covariates_embedding = self.cov_embed_net(covariates)
        x = self.graph_transformer(x, edge_index)
        age_pred = self.output_net(x, edge_index, batch, covariates_embedding)
        return age_pred

    def validation_step(self, batch, batch_idx):
        age_pred = self.forward(batch)
        loss = self.loss_fn(age_pred, batch.y)
        
        rmse = torch.sqrt(torch.mean((age_pred * self.config['std'] - self.config['mean'] - batch.y) ** 2)).item()
        self.val_rmse.append(rmse)
        # For MAE
        mae = torch.mean(torch.abs(age_pred * self.config['std'] - self.config['mean'] - batch.y)).item()
        self.val_mae.append(mae)
        
        # For Pearson correlation
        age_pred_cpu = age_pred.detach().cpu().numpy()
        batch_y_cpu = batch.y.detach().cpu().numpy()
        pearson_corr, _ = pearsonr(age_pred_cpu, batch_y_cpu)
        spearmanr_corr, _ = spearmanr(age_pred_cpu, batch_y_cpu)
        self.val_pearson.append(pearson_corr)
        self.val_spearman.append(spearmanr_corr)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # Average and log validation MAE and Pearson
        mean_rmse = np.mean(self.val_rmse)
        mean_mae = np.mean(self.val_mae)
        mean_pearson = np.mean(self.val_pearson)
        mean_spearman = np.mean(self.val_spearman)
        
        self.log('val_rmse', mean_rmse, on_epoch=True, sync_dist=True)
        self.log('val_mae', mean_mae, on_epoch=True, sync_dist=True)
        self.log('val_pearson', mean_pearson, on_epoch=True, sync_dist=True)
        self.log('val_spearman', mean_spearman, on_epoch=True, sync_dist=True)
        
        # Reset for the next epoch
        self.val_rmse = []
        self.val_mae = []
        self.val_pearson = []
        self.val_spearman = []

    def test_step(self, batch, batch_idx):
        age_pred = self.forward(batch)
        loss = self.loss_fn(age_pred, batch.y)
        
        rmse = torch.sqrt(torch.mean((age_pred * self.config['std'] - self.config['mean'] - batch.y) ** 2)).item()
        # For MAE
        mae = torch.mean(torch.abs(age_pred * self.config['std'] - self.config['mean'] - batch.y)).item()
        self.val_mae.append(mae)
        self.val_rmse.append(rmse)
        # For Pearson correlation
        age_pred_cpu = age_pred.detach().cpu().numpy()
        batch_y_cpu = batch.y.detach().cpu().numpy()
        pearson_corr, _ = pearsonr(age_pred_cpu, batch_y_cpu)
        spearmanr_corr, _ = spearmanr(age_pred_cpu, batch_y_cpu)
        self.val_pearson.append(pearson_corr)
        self.val_spearman.append(spearmanr_corr)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # Average and log validation MAE and Pearson
        mean_mae = np.mean(self.val_mae)
        mean_rmse = np.mean(self.val_rmse)
        mean_pearson = np.mean(self.val_pearson)
        mean_spearman = np.mean(self.val_spearman)
        
        self.log('test_rmse', mean_rmse, on_epoch=True, sync_dist=True)
        self.log('test_mae', mean_mae, on_epoch=True, sync_dist=True)
        self.log('test_pearson', mean_pearson, on_epoch=True, sync_dist=True)
        self.log('test_spearman', mean_spearman, on_epoch=True, sync_dist=True)
        
        # Reset for the next epoch
        self.val_rmse = []
        self.val_mae = []
        self.val_pearson = []
        self.val_spearman = []
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer








