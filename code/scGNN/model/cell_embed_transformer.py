import torch
from torch_scatter.segment_csr import gather_csr
from torch import nn
from linear_attention_transformer import LinearAttentionTransformer
from performer_pytorch import Performer

class CellEmbedTransformerLinearAttention(torch.nn.Module):
    def __init__(self, hidden, heads = 8, num_layers = 1, max_num_genes = 20000, config = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden = hidden
        self.heads = heads
        self.num_layers = num_layers
        self.expression_embed  = torch.nn.Embedding(config['bin_expression'] if config else 10, hidden)
        self.transformer = Performer(dim = hidden,
        depth = num_layers,
        heads = 8,
        dim_head = 64,
        causal = False, local_window_size= 512)
        self.linear = torch.nn.Linear(hidden, hidden)
        self.embedding = torch.nn.Embedding(max_num_genes, hidden)
        self.cls_embedding = torch.nn.Embedding(1, hidden)

    
    def forward(self, x, gene_type, batch):
        '''
        x: num_nodes * num_genes
        gene_type: batch * num_genes
        '''
        # batch * num_genes * hidden
        gene_type = gene_type.unsqueeze(0) if len(gene_type.shape) == 1 else gene_type
        gene_embed = self.embedding(gene_type)
        # num_nodes * num_genes * hidden
        gene_embed = gene_embed.index_select(0, batch)
        x = self.expression_embed(x) + gene_embed
        x = torch.cat([self.cls_embedding(torch.zeros(x.shape[0],1, dtype=torch.long, device=x.device)), x], dim=1)

        x = x[:, 0, :]
        return x


class CellEmbedTransformer(torch.nn.Module):
    '''
    Receive a X and gene type as input, and output the embedding of the cell type
    '''
    def __init__(self, hidden, heads = 8, num_layers = 1, max_num_genes = 20000, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden = hidden
        self.heads = heads
        self.num_layers = num_layers
        self.transformer = nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=hidden, dim_feedforward = 256, activation = 'gelu', 
                                                                           nhead=heads, batch_first=True) for _ in range(num_layers)])
        self.linear = torch.nn.Linear(hidden, hidden)
        self.embedding = torch.nn.Embedding(max_num_genes, hidden)
        self.cls_embedding = torch.nn.Embedding(1, hidden)

    
    def forward(self, x, gene_type, batch):
        '''
        x: num_nodes * num_genes
        gene_type: batch * num_genes
        '''
        # batch * num_genes * hidden
        gene_type = gene_type.unsqueeze(0) if len(gene_type.shape) == 1 else gene_type
        gene_embed = self.embedding(gene_type)
        # num_nodes * num_genes * hidden
        gene_embed = gene_embed.index_select(0, batch)

        x = x.unsqueeze(-1) + gene_embed
        x = torch.cat([self.cls_embedding(torch.zeros(x.shape[0],1, dtype=torch.long, device=x.device)), x], dim=1)
        for layer in self.transformer:
            x = layer(x)
        x = x[:, 0, :]
        return x

if __name__ == '__main__':
    x = torch.randn(100, 2000)
    gene_type = torch.randint(0, 10000, (2, 2000))
    batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    CellEmbedTransformer(128, 8)(x, gene_type, batch)
