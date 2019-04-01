import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_shape, output_shape, hid_shape, hid_num, activation='tanh'):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hid_shape = hid_shape
        self.hid_num = hid_num
        if activation == 'tanh':
            self.activation = [nn.Tanh, torch.tanh]
        elif activation == 'relu':
            self.activation = [nn.ReLU, torch.relu]
        else:
            raise Exception('unsupported activation type')

        layers = [nn.Linear(self.input_shape, self.hid_shape), self.activation[0]()]
        for _ in range(self.hid_num):
            layers.extend([nn.Linear(self.hid_shape, self.hid_shape), self.activation[0]()])
        layers.append(nn.Linear(self.hid_shape, self.output_shape))
        self.model = nn.Sequential(*layers)

    def forward(self, x, new_params=None):
        if new_params is None:
            return self.model(x)
        else:
            for i in range(self.hid_num + 1):
                x = F.linear(x, new_params['model.%d.weight' % (i * 2)], new_params['model.%d.bias' % (i * 2)])
                x = self.activation[1](x)
            x = F.linear(x, new_params['model.%d.weight' % ((self.hid_num + 1) * 2)], new_params['model.%d.bias' % ((self.hid_num + 1) * 2)])
            return x
            