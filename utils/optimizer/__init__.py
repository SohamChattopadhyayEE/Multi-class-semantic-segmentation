import torch

def get_optimizers(params, optimizer = 'Adam', lr = 0.001, mm = 0.9):
    if optimizer == 'SGD':
        return torch.optim.SGD(params, lr, mm)
    elif optimizer == 'RMSprop':
        return torch.optim.RMSprop(params, lr, mm)
    elif optimizer == 'Adam' :
        return torch.optim.Adam(params, lr, mm)
    else :
        return torch.optim.Adagrad(params, lr, mm)