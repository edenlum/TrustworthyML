import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint

def free_adv_train(model, data_tr, criterion, optimizer, lr_scheduler, \
                   eps, device, m=4, epochs=100, batch_size=128, dl_nw=10):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of 
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(data_tr,
                           batch_size=batch_size,
                           shuffle=True,
                           pin_memory=True,
                           num_workers=dl_nw)
                           

    # init delta (adv. perturbation) - FILL ME
    delta = torch.zeros(batch_size, 3, 32, 32, device=device)

    # total number of updates - FILL ME
    total_updates = epochs // m

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr)/batch_size))

    # train - FILLE ME
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(loader_tr, 0):
            # get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)
            b = inputs.shape[0] # real batch size in case last batch is different
            for j in range(m):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                x_adv = inputs + delta[:b]
                x_adv.requires_grad_()
                preds = model(x_adv)
                loss = criterion(preds, labels)
                loss.backward(retain_graph=True)
                optimizer.step()

                # update delta
                delta[:b] += eps * torch.sign(torch.autograd.grad(loss, x_adv)[0])
                delta = torch.clamp(delta, -eps, eps)
    
    # done
    return model


class SmoothedModel():
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size 
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        counts = torch.zeros(self.model.fc3.out_features)
        for i in range(n // batch_size):
            noise_shape = (batch_size,) + x.shape[1:]
            noise = torch.randn(noise_shape, device=x.device) * self.sigma
            preds = self.model(x + noise).detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            for p in preds:
                counts[p] += 1
        return counts

        
    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """
        
        # find prediction (top class c) - FILL ME
        counts0 = self._sample_under_noise(x, n0, batch_size)
        c_a = np.argmax(counts0)        
        
        # compute lower bound on p_c - FILL ME
        counts = self._sample_under_noise(x, n, batch_size)
        p_a = proportion_confint(counts[c_a].item(), n, alpha=2*alpha, method="beta")[0]

        if p_a > 0.5:
            c, radius = c_a, norm.ppf(p_a)*self.sigma
        else:
            c, radius = self.ABSTAIN, 0
        # done
        return c, radius
        

class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(self, model, dim=(1, 3, 32, 32), lambda_c=0.0005,
                 step_size=0.005, niters=2000):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask: 
        - trigger: 
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME
        mask = torch.rand(self.dim, device=device)
        mask.requires_grad_()
        trigger = torch.rand(self.dim, device=device)
        trigger.requires_grad_()

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME
        for i in range(self.niters):
            for x, y in data_loader:
                x = x.to(device)
                # override y to one-hot vector of target class c_t
                y = torch.zeros_like(y, device=device)
                y[:, c_t] = 1
                self.model.zero_grad()

                loss = self.loss_func(self.model((1 - mask) * x + mask * trigger), y)
                loss += self.lambda_c * torch.norm(mask)
                loss.backward()
                mask.data -= self.step_size * mask.grad
                trigger.data -= self.step_size * trigger.grad
                mask.data = torch.clamp(mask.data, 0, 1)
                trigger.data = torch.clamp(trigger.data, 0, 1)

        # done
        return mask, trigger
