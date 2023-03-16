import torch

def save_weight(path, epoch, model):
    state_dict = model.state_dict()
    data = {'epoch': epoch, 'state_dict': state_dict}
    torch.save(data, path)


def load_weight(path):
    pass