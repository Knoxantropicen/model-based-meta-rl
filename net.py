import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_shape, output_shape, hid_shape, hid_num, activation='tanh'):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hid_shape)
        self.fc2 = nn.Linear(hid_shape, hid_shape)
        self.fc3 = nn.Linear(hid_shape, output_shape)
        self.hid_num = hid_num
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise Exception('unsupported activation type')

    def forward(self, x):
        x = self.activation(self.fc1(x))
        for _ in range(self.hid_num):
            x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

