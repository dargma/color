

import torch
import torch.nn as nn

class Colornet(nn.Module):
    def __init__(self):
        super(Colornet, self).__init__()
        self.LeakyReLU = nn.LeakyReLU()	
        self.fc1 = nn.Linear(3,3)
        self.fc2 = nn.Linear(3,8)
        self.fc4 = nn.Linear(8,3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.fc4(x)

        return x
