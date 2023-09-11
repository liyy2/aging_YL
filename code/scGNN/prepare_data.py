# The file process the hut data set into a lmdb dataset
# import multiprocessing as mp
from importlib.metadata import metadata
from multiprocessing import Pool
from multiprocessing import Value
import os
import pickle
import pandas as pd
import lmdb
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.data import Data
# from AtomsToGraphs import AtomsToGraphs
from scGNN.data.process_data_utils import *
from scGNN.data.adata_utils import *
import torch
import argparse
from timeout_decorator import timeout

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning Aging Model Data Preprocessing')
    parser.add_argument('--num_genes', type=int, default=3000, help='number of hvg genes')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--cov_path', type=str, default='/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/PEC2_sample_metadata_processed.csv', help='path to covariate file')
    parser.add_argument('--out_path', type=str, default='/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/lmdb-test', help='path to output file')
    parser.add_argument('--file_path', type=str, default='/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices', help='path to input file')
    parser.add_argument('--embed_file', type=str, default='/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/gene-embeddings-Gene2Vec-200dim.csv', help='path to gene embedding file')
    parser.add_argument('--cell_type_encoder', type=str, default='/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/cell_type_encoder.pt', help='path to cell type encoder file')
    parser.add_argument('--min_cells', type=int, default=250, help='minimum number of cells')
    return parser.parse_args()
    
@timeout(900)
def process_individual(file_name, args, cell_type_encoder, gene_names_encoder):
    data = read_gzipped_bed_dask(file_name)
    adata = process_raw_matrix(data)
    adata = process_single_cell(adata, args.num_genes, gene_list_path = args.embed_file)
    x = torch.tensor(adata.X, dtype=torch.float)

    cell_type = adata.obs['cell_type'].values
    gene_names = adata.var['gene_symbols'].values
    # category encoding
    cell_type = cell_type_encoder.encode(cell_type)
    gene_type = gene_names_encoder.encode(gene_names)
    cell_type = torch.tensor(cell_type, dtype=torch.long)
    gene_type = torch.tensor(gene_type, dtype=torch.long)
    edge_index = neighbor_graph(adata)
    covariates = load_covariates(file_name, cov_path=args.cov_path)
    y = covariates['Age_death']
    covariates = covariates.drop(['Age_death'])
    covariates = torch.tensor(covariates.values, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, covariates=covariates.unsqueeze(0), 
                y = y, cell_type=cell_type, gene_type=gene_type)
    
    # some data is not valid without gene type
    data.gene_type.max()
    return data

        

def write_images_to_lmdb(mp_arg):
    db_path, file_list, idx, pid, args = mp_arg
    cell_type_encoder = RuntimeCategoryEncoder()
    cell_type_encoder.load(args.cell_type_encoder)
    cell_type_encoder.static = True
    gene_names_encoder = RuntimeCategoryEncoder(embed_file=args.embed_file)
    gene_names_encoder.static = True
    black_list = [
            '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Girgenti-multiome/RT00389N-annotated_matrix.txt.gz', 
            '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/UCLA-ASD/5297-annotated_matrix.txt.gz',
            '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Ma_et_al/HSB189-annotated_matrix.txt.gz',
            '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/Ma_et_al/HSB340-annotated_matrix.txt.gz',
            '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/UCLA-ASD/5297-annotated_matrix.txt.gz',
            '/gpfs/gibbs/pi/gerstein/jjl86/project/aging_YL/snrna_expr_matrices/IsoHuB/HSB189-annotated_matrix.txt.gz']
    with lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    ) as db:
        for index in trange(len(file_list), desc=f"Worker {pid}" ):
            try:
                if file_list[index] in black_list:
                    continue
                tqdm.write(f"Worker {pid} processing {file_list[index]}")
                data = process_individual(file_list[index], args, cell_type_encoder, gene_names_encoder)
                txn = db.begin(write=True)
                txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
                idx += 1
                txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
                txn.commit()
            except Exception as e:
                print(f'{file_list[index]} is not valid')
                print(e)
    return [0]



def main():
    args = parse_args()
    num_workers = args.num_workers
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    db_paths = [
        os.path.join(out_path, "data.%04d.lmdb" % i)
        for i in range(num_workers)
    ]
    file_path = args.file_path
    cohort = os.listdir(file_path)
    # list the file in each cohort
    file_list = [os.listdir(os.path.join(file_path, cohort)) for cohort in cohort]
    file_list = [[os.path.join(file_path, cohort, f) for f in file if f.endswith('.txt.gz')] for cohort, file in zip(cohort, file_list)]
    file_list = [file for cohort in file_list for file in cohort]
    file_list = np.array(file_list).flatten()     
    chunked_txt_files = np.array_split(file_list, num_workers)
    idx_all = [0] * num_workers

    pool = Pool(num_workers)
    mp_args = [
        (
            db_paths[i],
            chunked_txt_files[i],
            idx_all[i],
            i,
            args,
        )
        for i in range(num_workers)
    ]
    
    result = pool.map_async(write_images_to_lmdb, mp_args)
    
    # Get results
    processed_data = result.get()
    
    
if __name__ == "__main__":
    main()