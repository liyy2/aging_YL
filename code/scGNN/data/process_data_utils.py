import gzip
import pandas as pd
import scanpy as sc
import pandas as pd
from scipy.sparse import find
import torch
from torch_geometric.data import Data
import os

__cov_cat_list__ = [12, 2, 10, 6]


def read_gzipped_bed(file_path):
    with gzip.open(file_path, 'rt') as f:
        data = pd.read_csv(f, sep="\t")
    return data

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





    
