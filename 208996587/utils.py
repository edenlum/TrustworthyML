import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id<0 or cnn_id>2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model

class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """
    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    model.to(device)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            correct += (pred.argmax(dim=1) == y).sum().item()
    return correct / len(data_loader.dataset)


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    x_advs = []
    labels = []
    from tqdm import tqdm

    for x, y in tqdm(data_loader):
        x, y = x.to(device), y.to(device)
        if targeted:
            y = (y + torch.randint(1, n_classes, size=(len(y),)).to(device)) % n_classes
        x_adv = attack.execute(x, y, targeted=targeted)
        x_advs.append(x_adv)
        labels.append(y)
    return torch.cat(x_advs), torch.cat(labels)

def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    x_advs = []
    labels = []
    num_queries = []
    
    from tqdm import tqdm

    for x, y in tqdm(data_loader):
        x, y = x.to(device), y.to(device)
        if targeted:
            y = (y + torch.randint(1, n_classes, size=(len(y),)).to(device)) % n_classes
        x_adv, queries = attack.execute(x, y, targeted=targeted)
        x_advs.append(x_adv)
        labels.append(y)
        num_queries.append(queries)
    return torch.cat(x_advs), torch.cat(labels), torch.cat(num_queries)

def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_adv, y),
        batch_size=batch_size
    )
    success = 0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        if targeted:
            success += (y_pred.argmax(dim=1) == y_batch).sum().item()
        else:
            success += (y_pred.argmax(dim=1) != y_batch).sum().item()
    return success / len(y)
        

def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    return ''.join('{:08b}'.format(c) for c in struct.pack('!f', num))

def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    return struct.unpack('!f', struct.pack('!I', int(binary, 2)))[0]

def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    binary_w = binary(w)
    idx = np.random.randint(0, 32)
    binary_w_list = list(binary_w)
    binary_w_list[idx] = '1' if binary_w[idx] == '0' else '0'
    binary_w = ''.join(binary_w_list)
    return float32(binary_w), idx
