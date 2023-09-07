import pandas as pd
import scanpy as sc
import torch
def process_raw_matrix(df):
    # Assuming `df` is your DataFrame
    # df = pd.read_csv("your_file.csv")

    # Extract the gene names from the 'featurekey' column
    gene_names = df['featurekey'].values

    # Drop the 'featurekey' column to only keep the count data
    count_data = df.drop(['featurekey'], axis=1)

    # Transpose the DataFrame so that rows are samples and columns are genes
    # This is a common format that Scanpy expects
    count_data = count_data.T

    # Create a list of cell types based on column names (which are now row indices after the transpose)
    cell_types = [name.split('.')[0] for name in count_data.index]  # Assumes the cell type is the prefix before the '.'
    cell_types = pd.Series(cell_types, index=count_data.index)  # Convert to a Pandas Series

    # Convert the DataFrame to an AnnData object
    adata = sc.AnnData(X=count_data, var={"gene_symbols": gene_names})
    

    # Add cell type as metadata
    adata.obs['cell_type'] = cell_types
    return adata 



import scanpy as sc

def process_single_cell(adata, num_genes = 500):
    # Filter out genes that are detected in less than 3 cells
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    # Normalize the data to 10,000 reads per cell, so that counts become comparable among cells
    sc.pp.normalize_total(adata, target_sum=1e4)

    # Logarithmize the data
    sc.pp.log1p(adata)
    ### filter cells
    # Filter out cells that have more than 10% mitochondrial genes expressed
    # Calculate the percentage of mitochondrial genes expressed
    
    
    # # Identify Highly Variable Genes
    sc.pp.highly_variable_genes(adata)

    # Sort genes by dispersion values in descending order
    hvg_sorted = adata.var.sort_values(by="dispersions", ascending=False).index

    # Take the top 2000 highly variable genes
    top_hvg = hvg_sorted[:num_genes]

    # Subset the data to include only the top 2000 HVGs
    adata = adata[:, top_hvg]

    # Scale the data to unit variance and zero mean
    sc.pp.scale(adata, max_value=10)

    # Perform PCA
    sc.tl.pca(adata, svd_solver='arpack')
    actual_n_pcs = min(40, adata.obsm['X_pca'].shape[1])
    # Perform nearest neighbor search
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=actual_n_pcs)

    # Uncomment other downstream methods if needed
    # Perform UMAP
    # sc.tl.umap(adata)
    # # Perform Louvain clustering
    # sc.tl.louvain(adata, resolution=0.3)
    # # Perform Leiden clustering
    # sc.tl.leiden(adata, resolution=0.3)
    # # Perform tSNE
    # sc.tl.tsne(adata)
    # # Perform Diffusion Map
    # sc.tl.diffmap(adata)

    return adata

def neighbor_graph(adata):
    '''
    return a neighbor graph
    '''
    from scipy.sparse import find

    # Access the KNN graph (stored as a sparse matrix)
    knn_graph = adata.obsp['connectivities']

    # Extract edge indices
    source, target, weight = find(knn_graph)
    import torch
    source = torch.tensor(source, dtype=torch.long)
    target = torch.tensor(target, dtype=torch.long)
    edge_index = torch.stack([source, target], dim=0)
    return edge_index

