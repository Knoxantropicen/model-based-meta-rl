import numpy as np
import torch
from torch.distributions.normal import Normal
from tools.utils import cuda_tensor


def adam_update(p, loss, optimizer):
    grad = torch.autograd.grad(loss, p, retain_graph=True)[0]
    group = optimizer.param_groups[0]
    state = optimizer.state[p]
    amsgrad = group['amsgrad']
    if len(state) == 0:
        state['step'] = 0
        state['exp_avg'] = torch.zeros_like(p.data)
        state['exp_avg_sq'] = torch.zeros_like(p.data)
        if amsgrad:
            state['max_exp_avg_sq'] = torch.zeros_like(p.data)
    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    if amsgrad:
        max_exp_avg_sq = state['max_exp_avg_sq']
    beta1, beta2 = group['betas']
    if group['weight_decay'] != 0:
        grad.add_(group['weight_decay'], p.data)
    exp_avg.mul_(beta1).add_(1 - beta1, grad)
    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
    state['step'] += 1
    if amsgrad:
        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = max_exp_avg_sq.sqrt().add_(group['eps'])
    else:
        denom = exp_avg_sq.sqrt().add_(group['eps'])
    bias_correction1 = 1 - beta1 ** state['step']
    bias_correction2 = 1 - beta2 ** state['step']
    step_size = group['lr'] * np.sqrt(bias_correction2) / bias_correction1
    p.data.addcdiv_(-step_size, exp_avg, denom)


class Loss:
    def __init__(self, *args, **kwargs):
        self.optimizer = None

    def get_loss(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        pass

    def zero_grad(self, *args, **kwargs):
        if self.optimizer:
            self.optimizer.zero_grad()

    def step(self, *args, **kwargs):
        if self.optimizer:
            self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict() if self.optimizer else None
 
    def load_state_dict(self, state_dict):
        if self.optimizer:
            self.optimizer.load_state_dict(state_dict)
        

class MSELoss(Loss):
    def __init__(self, loss_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_scale = loss_scale

    def get_loss(self, *args, **kwargs):
        return self.loss_scale * torch.sqrt(torch.sum(torch.pow(args[0] - args[1], 2)))


class NLLLoss(Loss):
    def __init__(self, loss_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_scale = loss_scale
        self.std = cuda_tensor(0.1, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.std])

    def get_loss(self, *args, **kwargs):
        normal = Normal(loc=args[0], scale=self.std)
        return self.loss_scale * torch.sum(-normal.log_prob(args[1]))

    def update(self, loss):
        adam_update(self.std, loss, self.optimizer)
        self.std.data = torch.max(self.std.data, torch.zeros_like(self.std.data))

    def step(self, *args, **kwargs):
        super().step(*args, **kwargs)
        self.std.data = torch.max(self.std.data, torch.zeros_like(self.std.data))


loss_func = {
    'mse': MSELoss,
    'nll': NLLLoss,
}
