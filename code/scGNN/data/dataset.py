import torch
import os
from scGNN.data.process_data_utils import *
from scGNN.data.adata_utils import *
from torch_geometric.data import Data
from torch.utils.data import random_split
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader, NeighborSampler
from tqdm import trange
import logging
# from torch import neighbor_graph

class RuntimeCategoryEncoder:
    def __init__(self):
        self.category_to_code = {}
        self.code_to_category = {}
        self.next_code = 0
    
    def encode(self, categories):
        codes = []
        for category in categories:
            if category not in self.category_to_code:
                self.category_to_code[category] = self.next_code
                self.code_to_category[self.next_code] = category
                self.next_code += 1
            codes.append(self.category_to_code[category])
        return codes



class NodeSampler:
    def __init__(self, sample_ratio=0.2):
        self.sample_ratio = sample_ratio

    def sample_edges(self, edge_index, sample_idx):
        # Create a tensor to store the remapped indices
        remap_tensor = torch.full((edge_index.max().item() + 1,), -1, dtype=torch.long)
        
        # Fill the remapped indices
        remap_tensor[sample_idx] = torch.arange(len(sample_idx))
        
        # Filter and remap the edge_index
        mask = (remap_tensor[edge_index[0]] >= 0) & (remap_tensor[edge_index[1]] >= 0)
        edge_index = edge_index[:, mask]
        edge_index = remap_tensor[edge_index]
        
        return edge_index

    def __call__(self, data):
        n = data.x.shape[0]
        n_sample = int(n * self.sample_ratio)
        sample_idx = torch.randperm(n)[:n_sample]
        
        # Subsample node features
        data.x = data.x[sample_idx]
        if hasattr(data, 'cell_type'):  # Make sure 'cell_type' attribute exists
            data.cell_type = data.cell_type[sample_idx]

        # Subsample and remap edge indices
        if hasattr(data, 'edge_index'):  # Make sure 'edge_index' attribute exists
            data.edge_index = self.sample_edges(data.edge_index, sample_idx)

        return data




class SingleCellDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, transforms  = [NodeSampler(0.005)], config = None):
        super(SingleCellDataset, self).__init__()
        # list all the cohort
        self.cohort = os.listdir(file_path)
        # list the file in each cohort
        self.file_list = [os.listdir(os.path.join(file_path, cohort)) for cohort in self.cohort]
        self.file_list = [[os.path.join(file_path, cohort, f) for f in file if f.endswith('.txt.gz')] for cohort, file in zip(self.cohort, self.file_list)]
        # flatten the list
        self.file_list = [file for cohort in self.file_list for file in cohort]
        self.cov_path = '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/PEC2_sample_metadata_processed.csv'
        self.cell_type_encoder = RuntimeCategoryEncoder()
        self.gene_names_encoder = RuntimeCategoryEncoder()
        self.transform = transforms
        self.config = config

    def prepare_data(self):
        logging.info('Start preparing data')
        for index in trange(len(self)):
            data = read_gzipped_bed(self.file_list[index])
            adata = process_raw_matrix(data)
            adata = process_single_cell(adata, self.config['num_genes'] if self.config else 500)
            x = torch.tensor(adata.X, dtype=torch.float)
            cell_type = adata.obs['cell_type'].values
            gene_names = adata.var['gene_symbols'].values
            # category encoding
            cell_type = self.cell_type_encoder.encode(cell_type)
            gene_type = self.gene_names_encoder.encode(gene_names)
            cell_type = torch.tensor(cell_type, dtype=torch.long)
            gene_type = torch.tensor(gene_type, dtype=torch.long)
            edge_index = neighbor_graph(adata)
            covariates = load_covariates(self.file_list[index], self.cov_path)
            y = covariates['Age_death']
            covariates = covariates.drop(['Age_death'])
            covariates = torch.tensor(covariates.values, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, covariates=covariates.unsqueeze(0), 
                        y = y, cell_type=cell_type, gene_type=gene_type)
            torch.save(data, self.file_list[index].replace('.txt.gz', '.pt'))


    def __getitem__(self, index):
        if os.path.exists(self.file_list[index].replace('.txt.gz', '.pt')):
            data = torch.load(self.file_list[index].replace('.txt.gz', '.pt'))
            for transform in self.transform:
                data = transform(data)
            return data
        else:
            raise FileNotFoundError(f'{self.file_list[index].replace(".txt.gz", ".pt")} does not exist')

    def __len__(self):
        return len(self.file_list)

class SingleCellDataModule(LightningDataModule):
    def __init__(self, file_path, batch_size=2, config = None):
        super(SingleCellDataModule, self).__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.config = config
        self.prepare_data() if config['preprocess'] else None
        self.setup()
        

    def prepare_data(self):
        # Optional method to download or prepare your data here
        full_dataset = SingleCellDataset(self.file_path, config=self.config)
        full_dataset.prepare_data()
        pass

    def setup(self, seed = 42, stage=None):
        # Assign train/val datasets
        full_dataset = SingleCellDataset(self.file_path, config=self.config)
        n_train = int(len(full_dataset) * 0.7)
        n_val = int(len(full_dataset) * 0.1)
        n_test = len(full_dataset) - n_train - n_val
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)


if __name__ == "__main__":
    dataset = SingleCellDataModule('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices')
    next(iter(dataset.train_dataloader()))
    print(dataset[0])