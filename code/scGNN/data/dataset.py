import torch
import os
from scGNN.data.process_data_utils import *
from scGNN.data.adata_utils import *
from torch_geometric.data import Data, DataListLoader
from torch.utils.data import random_split
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader


class SingleCellDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(SingleCellDataset, self).__init__()
        # list all the cohort
        self.cohort = os.listdir(file_path)
        # list the file in each cohort
        self.file_list = [os.listdir(os.path.join(file_path, cohort)) for cohort in self.cohort]
        self.file_list = [[os.path.join(file_path, cohort, f) for f in file] for cohort, file in zip(self.cohort, self.file_list)]
        # flatten the list
        self.file_list = [file for cohort in self.file_list for file in cohort]
        self.cov_path = '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/PEC2_sample_metadata_processed.csv'

    def __getitem__(self, index):
        data = read_gzipped_bed(self.file_list[index])
        adata = process_raw_matrix(data)
        adata = process_single_cell(adata)
        x = torch.tensor(adata.X, dtype=torch.float)
        cell_type = adata.obs['cell_type'].values 
        # category encoding
        cell_type = pd.Categorical(cell_type).codes
        cell_type = torch.tensor(cell_type, dtype=torch.long)
        edge_index = neighbor_graph(adata)
        covariates = load_covariates(self.file_list[index], self.cov_path)
        y = covariates['Age_death']
        covariates = covariates.drop(['Age_death'], axis=1)
        covariates = torch.tensor(covariates.values, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, covariates=covariates, y = y, cell_type=cell_type)

    def __len__(self):
        return len(self.file_list)

class SingleCellDataModule(LightningDataModule):
    def __init__(self, file_path, batch_size=32):
        super(SingleCellDataModule, self).__init__()
        self.file_path = file_path
        self.batch_size = batch_size

    def prepare_data(self):
        # Optional method to download or prepare your data here
        pass

    def setup(self, stage=None):
        # Assign train/val datasets
        full_dataset = SingleCellDataset(self.file_path)
        n_train = int(len(full_dataset) * 0.7)
        n_val = int(len(full_dataset) * 0.1)
        n_test = len(full_dataset) - n_train - n_val
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [n_train, n_val, n_test])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    dataset = SingleCellDataset('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices')
    print(dataset[0])