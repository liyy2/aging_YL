import torch
import torch_scatter

class CovEmbedNet(torch.nn.Module):
    '''
    Receive a X as input, and output the embedding of the covariates
    '''
    def __init__(self, category_list, hidden):
        super().__init__()
        self.category_list = category_list
        self.embedding_list = torch.nn.ModuleList([torch.nn.Embedding(category, 10) for category in category_list])
        self.linear = torch.nn.Linear(10 * len(category_list), hidden)
    
    def forward(self, cov):
        cov_embedding = [embedding(cov[:, i]) for i, embedding in enumerate(self.embedding_list)]
        cov_embedding = torch.cat(cov_embedding, dim=1)
        cov_embedding = self.linear(cov_embedding)
        return cov_embedding