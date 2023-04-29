import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
      """
      Executes the attack on a batch of samples x. y contains the true labels 
      in case of untargeted attacks, and the target labels in case of targeted 
      attacks. The method returns the adversarially perturbed samples, which
      lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
      performs random initialization and early stopping, depending on the 
      self.rand_init and self.early_stop flags.
      """
      if self.rand_init:
            x_adv = x + torch.empty_like(x).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
      else:
            x_adv = x.clone().detach()
      for i in range(self.n):
            x_adv.requires_grad_()
            with torch.enable_grad():
                  logits = self.model(x_adv)
                  loss = self.loss_func(logits, y) # tensor of shape (batch_size,)
                  grad = torch.autograd.grad(loss.sum(), x_adv)[0]

            # early stop
            success = (logits.argmax(1).eq(y) if targeted else logits.argmax(1).ne(y)).all()
            if self.early_stop and success:
                  break
            
            # compute step
            if targeted:
                  x_adv = x_adv - self.alpha * torch.sign(grad)
            else:
                  x_adv = x_adv + self.alpha * torch.sign(grad)
            
            # projection step
            x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

      assert (x_adv - x).abs().max() <= self.eps + 1e-5      
      return x_adv



class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
      """
      Executes the attack on a batch of samples x. y contains the true labels 
      in case of untargeted attacks, and the target labels in case of targeted 
      attacks. The method returns:
      1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
      and [x-eps, x+eps].
      2- A vector with dimensionality len(x) containing the number of queries for
      each sample in x.
      """
      from tqdm import tqdm

      if self.rand_init:
            x_adv = x + torch.empty_like(x).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)
      else:
            x_adv = x.clone().detach()
      grad_avg = torch.zeros_like(x)
      queries = torch.zeros(len(x), dtype=torch.int32)
      nosuccess = torch.ones(len(x), dtype=torch.bool)
      for i in tqdm(range(self.n)):
            print(nosuccess.sum())
            grad = torch.zeros_like(x[nosuccess])
            for _ in range(self.k):
                  noise = torch.randn_like(x[nosuccess])
                  x_pos = torch.clamp(x_adv[nosuccess] + self.sigma * noise, 0, 1)
                  x_neg = torch.clamp(x_adv[nosuccess] - self.sigma * noise, 0, 1)
                  with torch.no_grad():
                        logits_pos = self.model(x_pos)
                        logits_neg = self.model(x_neg)
                        loss_pos = self.loss_func(logits_pos, y[nosuccess]) 
                        loss_neg = self.loss_func(logits_neg, y[nosuccess])
                  loss = loss_pos - loss_neg
                  grad += loss.unsqueeze(1).unsqueeze(2).unsqueeze(3) * noise
                  
            queries[nosuccess] += 2 * self.k # 2 queries per iteration
            grad /= (2 * self.k * self.sigma)
            grad_avg[nosuccess] = self.momentum * grad_avg[nosuccess] + (1 - self.momentum) * grad
            
            if targeted:
                  x_adv[nosuccess] += - self.alpha * torch.sign(grad_avg[nosuccess])
            else:
                  x_adv[nosuccess] += self.alpha * torch.sign(grad_avg[nosuccess])
            
            # projection step
            x_adv = torch.clamp(x_adv, x - self.eps, x + self.eps)
            x_adv = torch.clamp(x_adv, 0, 1)

            # if the attack goal is met on a sample, then stop perturbing it by removing it from the batch
            logits = self.model(x_adv)
            success = (logits.argmax(1).eq(y) if targeted else logits.argmax(1).ne(y))
            nosuccess = ~success
            if self.early_stop and success.all():
                  break 

      assert (x_adv - x).abs().max() <= self.eps + 1e-5
      return x_adv, torch.full((len(x),), i * (2 * self.k + 1))



class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        pass  # FILL ME
