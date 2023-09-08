import torch
import os
from scGNN.data.process_data_utils import *
from scGNN.data.adata_utils import *
from torch_geometric.data import Data
from torch.utils.data import random_split
from torch_geometric.utils import add_self_loops, to_undirected
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from tqdm import trange
import logging
# from torch import neighbor_graph





class SingleCellDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, config = None):
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
        self.config = config
        self.transforms = [NodeSampler(config['node_sampling_ratio'], config['min_node_num'], config['max_node_num']), 
                           ExpressionBinner(config['bin_expression'])]
        self.QC()
        self.file_list = [file for file in self.file_list if file not in self.black_list]


    def QC(self):
        if os.path.exists('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/black_list_preprocess.pt'):
            black_list = torch.load('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/black_list_preprocess.pt')
        # if os.path.exists('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/black_list.pt'):
        #     self.black_list = torch.load('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/black_list.pt')
        # else:

        # black_list = torch.load('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/black_list.pt') if os.path.exists(
        #     '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/black_list.pt') else [
        #         '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Girgenti-multiome/RT00389N-annotated_matrix.txt.gz', 
        #         '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/UCLA-ASD/5297-annotated_matrix.txt.gz',
        #         '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Ma_et_al/HSB189-annotated_matrix.txt.gz',
        #         '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Ma_et_al/HSB340-annotated_matrix.txt.gz',
        #         '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/UCLA-ASD/5297-annotated_matrix.txt.gz',
        #         '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/IsoHuB/HSB189-annotated_matrix.txt.gz']            
        

        for index in trange(len(self.file_list)):
            if self.file_list[index].replace('.txt.gz', '.pt') in black_list:
                continue
            if os.path.exists(self.file_list[index].replace('.txt.gz', '.pt')):
                data = torch.load(self.file_list[index].replace('.txt.gz', '.pt'))
                # print(f'{data.x.shape[0]} cells in {self.file_list[index]}')
                try:
                    data.gene_type
                except:
                    print(f'{self.file_list[index]} does not have gene_type')
                    black_list.append(self.file_list[index])
                    continue
                if data.x.shape[0] < 150:
                    black_list.append(self.file_list[index])
                    continue
            else:
                black_list.append(self.file_list[index])
        
        torch.save(black_list, '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/black_list.pt')
        self.black_list = black_list


    def prepare_data(self):
        pass
    # def prepare_data(self):
    #     logging.info('Start preparing data')
    #     black_list = [
    #             '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Girgenti-multiome/RT00389N-annotated_matrix.txt.gz', 
    #             '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/UCLA-ASD/5297-annotated_matrix.txt.gz',
    #             '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Ma_et_al/HSB189-annotated_matrix.txt.gz',
    #             '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Ma_et_al/HSB340-annotated_matrix.txt.gz',
    #             '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/UCLA-ASD/5297-annotated_matrix.txt.gz',
    #             '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/IsoHuB/HSB189-annotated_matrix.txt.gz']  
    #     for index in trange(len(self.file_list)):
    #         try:
    #             if self.file_list[index].replace('.txt.gz', '.pt') in black_list:
    #                 continue
    #             if os.path.exists(self.file_list[index].replace('.txt.gz', '.pt')):
    #                 os.remove(self.file_list[index].replace('.txt.gz', '.pt'))
    #             data = read_gzipped_bed(self.file_list[index])
    #             adata = process_raw_matrix(data)
    #             adata = process_single_cell(adata, self.config['num_genes'] if self.config else 500)
    #             x = torch.tensor(adata.X, dtype=torch.float)
    #             cell_type = adata.obs['cell_type'].values
    #             gene_names = adata.var['gene_symbols'].values
    #             # category encoding
    #             cell_type = self.cell_type_encoder.encode(cell_type)
    #             gene_type = self.gene_names_encoder.encode(gene_names)
    #             cell_type = torch.tensor(cell_type, dtype=torch.long)
    #             gene_type = torch.tensor(gene_type, dtype=torch.long)
    #             edge_index = neighbor_graph(adata)
    #             covariates = load_covariates(self.file_list[index], self.cov_path)
    #             y = covariates['Age_death']
    #             covariates = covariates.drop(['Age_death'])
    #             covariates = torch.tensor(covariates.values, dtype=torch.long)
    #             y = torch.tensor(y, dtype=torch.float)
    #             data = Data(x=x, edge_index=edge_index, covariates=covariates.unsqueeze(0), 
    #                         y = y, cell_type=cell_type, gene_type=gene_type)

    #             torch.save(data, self.file_list[index].replace('.txt.gz', '.pt'))
    #         except:
    #             print(f'{self.file_list[index]} is not valid')
    #             black_list.append(self.file_list[index])
    #             continue
    #         finally:
    #             self.cell_type_encoder.save('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/cell_type_encoder.pt')
    #             self.gene_names_encoder.save('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/gene_names_encoder.pt')
    #             torch.save(black_list, '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/black_list_preprocess.pt')
                


    def __getitem__(self, index):
        if os.path.exists(self.file_list[index].replace('.txt.gz', '.pt')):
            data = torch.load(self.file_list[index].replace('.txt.gz', '.pt'))
            for transform in self.transforms:
                data = transform(data)
            data.gene_type = data.gene_type.unsqueeze(0) if len(data.gene_type.shape) == 1 else data.gene_type
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
        self.setup()
        

    def prepare_data(self):
        # Optional method to download or prepare your data here
        full_dataset = SingleCellDataset(self.file_path, config=self.config)
        full_dataset.prepare_data()
        pass

    def setup(self, seed = 42, stage=None):
        # Assign train/val datasets
        full_dataset = SingleCellDataset(self.file_path, config=self.config)
        n_train = int(len(full_dataset) * 0.8)
        n_val = int(len(full_dataset) * 0.2)
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