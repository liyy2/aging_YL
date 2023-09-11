import gzip
import pandas as pd
import scanpy as sc
import pandas as pd
from scipy.sparse import find
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected
import os
import dask.dataframe as dd

__cov_cat_list__ = [12, 2, 10, 6]


def read_gzipped_bed(file_path):
    with gzip.open(file_path, 'rt') as f:
        data = pd.read_csv(f, sep="\t")
    return data

def read_gzipped_bed_dask(file_path):
    data = dd.read_csv(file_path, sep="\t", compression="gzip", blocksize=None)
    return data.compute()

def load_covariates(file_path, cov_path):
    cov = pd.read_csv(cov_path, sep =',' )
    # Assuming cov is your DataFrame
    cov.set_index('Individual_ID', inplace=True)
    cov['Age_death'] = cov['Age_death'].replace('89+', '90')
    cov['Age_death'] = cov['Age_death'].replace('90+', '90')
    cov['Age_death'] = cov['Age_death'].astype(float)
    # cov_encoded = pd.get_dummies(cov, 
    #     columns=['Cohort', 'Biological_Sex', 'Disorder', 
    #     '1000G_ancestry'])  
    cov_encoded = cov
    # label encode the covariates beside age
    for col in cov_encoded.columns:
        if col != 'Age_death':
            cov_encoded[col] = cov_encoded[col].astype('category').cat.codes
    file_path = os.path.basename(file_path)
    id = file_path.split('-')[0]
    return cov_encoded.loc[id]

class RuntimeCategoryEncoder:
    def __init__(self, embed_file = None, static = False):
        self.category_to_code = {}
        self.code_to_category = {}
        self.next_code = 0
        self.static = static
        if embed_file:
            file = pd.read_csv(embed_file)
            self.category_to_code = dict(zip(file['canonical_symbol'].to_list(), file.index.tolist()))
            self.code_to_category = dict(zip(file.index.tolist(), file['canonical_symbol'].to_list()))
            self.next_code = len(file['canonical_symbol'])

    
    def encode(self, categories):
        if isinstance(categories, str):
            categories = [categories]
        codes = []
        for category in categories:
            if category not in self.category_to_code:
                if self.static:
                    raise Exception('CategoryEncoder is static, but category {} is not in the dictionary'.format(category))
                self.category_to_code[category] = self.next_code
                self.code_to_category[self.next_code] = category
                self.next_code += 1
            codes.append(self.category_to_code[category])
        return codes

    def decode(self, codes):
        categories = []
        for code in codes:
            categories.append(self.code_to_category[code])
        return categories
    
    def save(self, path):
        torch.save({
            'category_to_code': self.category_to_code,
            'code_to_category': self.code_to_category,
            'next_code': self.next_code
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.category_to_code = checkpoint['category_to_code']
        self.code_to_category = checkpoint['code_to_category']
        self.next_code = checkpoint['next_code']


class NodeSampler:
    def __init__(self, sample_ratio=0.2, min_cells=300, max_cells=450):
        self.sample_ratio = sample_ratio
        self.min_cells = min_cells
        self.max_cells =  max_cells

    def sample_edges(self, edge_index, sample_idx, num_cells):
        # Create a tensor to store the remapped indices
        remap_tensor = torch.full((num_cells,), -1, dtype=torch.long)
        
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
        if self.min_cells > n_sample:
            n_sample = self.min_cells
            if n < n_sample:
                n_sample = n
        if self.max_cells < n_sample:
            n_sample = self.max_cells
        sample_idx = torch.randperm(n)[:n_sample]
        
        # Subsample node features
        data.x = data.x[sample_idx]
        if hasattr(data, 'cell_type'):  # Make sure 'cell_type' attribute exists
            data.cell_type = data.cell_type[sample_idx]

        # Subsample and remap edge indices
        if hasattr(data, 'edge_index'):  # Make sure 'edge_index' attribute exists
            data.edge_index = self.sample_edges(data.edge_index, sample_idx, n)
            data.edge_index = to_undirected(data.edge_index)
            data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=n_sample)

        return data

def bin_values(matrix, k):
    """
    Bin expression values for each cell into k classes using quantiles.
    :param matrix: torch.Tensor of size (num_cells, num_genes)
    :param k: number of bins/classes
    :return: torch.Tensor of size (num_cells, num_genes) with binned values
    """
    # Compute quantiles for each cell simultaneously
    quantiles = torch.quantile(matrix, torch.linspace(0, 1, steps=k+1)[1:-1], dim=1).transpose(0, 1)

    # Expand dimensions for broadcasting
    expanded_matrix = matrix.unsqueeze(-1)
    expanded_quantiles = quantiles.unsqueeze(1)

    # Compute binned values for all cells and genes at once
    binned_values = (expanded_matrix > expanded_quantiles).sum(dim=-1)
    binned_values = binned_values.to(dtype=torch.long)

    return binned_values



class ExpressionBinner:
    def __init__(self, k):
        self.k = k

    def __call__(self, data):
        if self.k > 0:
            data.x = bin_values(data.x, self.k)
        return data

# Sample data: expression_matrix with shape (num_cells, num_genes)
# Initialize with random values as an example

# if __name__ == '__main__':
#     num_cells = 100
#     num_genes = 50
#     expression_matrix = torch.rand(num_cells, num_genes)
#     k = 5  # number of bins/classes

#     binned_matrix = bin_values(expression_matrix, k)
#     print(binned_matrix[0], expression_matrix[0])


if __name__ == '__main__':
    # data = read_gzipped_bed_dask('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/CMC/CMC_MSSM_002-annotated_matrix.txt.gz')
    data = read_gzipped_bed_dask('/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Ma_et_al/HSB628-annotated_matrix.txt.gz', )
    
    print(data.iloc[:, 0:10])




    
