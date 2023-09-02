import torch

class BaseLoss(torch.nn.Module):
    def __init__(self, mean_train, std_train):
        super(BaseLoss, self).__init__()
        self.base_loss = None
        self.mean_train = mean_train
        self.std_train = std_train

    def forward(self, pred, true):
        pred_rescaled = pred * self.std_train + self.mean_train

        loss_mean = self.base_loss(pred_rescaled, true)
        
        return loss_mean # Replace with your actual formula



class MSE(BaseLoss):
    def __init__(self, mean_train, std_train):
        super(MSE, self).__init__(mean_train, std_train)
        self.base_loss = torch.nn.MSELoss()

class Huber(BaseLoss):
    def __init__(self, mean_train, std_train):
        super(Huber, self).__init__(mean_train, std_train)
        self.base_loss = torch.nn.SmoothL1Loss()