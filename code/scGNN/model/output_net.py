import torch
import torch_scatter

class OutputNet(torch.nn.Module):
    '''
    OutputNet for age prediction, recieve the mean of the age and std of the age as input
    '''

    def __init__(self, hidden_dim, age_mean, age_std, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer('age_mean', age_mean)
        self.register_buffer('age_std', age_std)
        self.linear = torch.nn.Linear(2 * hidden_dim, 1)
        self.reduce = 'mean'
    
    def forward(self, x, edge_index, batch, covariates_embedding):
        individual_rep = torch_scatter.scatter(x, batch, dim=0, reduce=self.reduce)
        individual_rep = torch.cat([individual_rep, covariates_embedding], dim=1)
        age_pred = self.linear(individual_rep)
        return age_pred





