import torch
import torch.nn as nn


class GestureFNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(GestureFNN, self).__init__()
        # Linear function 1: 126 --> 84
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim_1)

        # Non-linearity 1
        self.sigmoid = torch.nn.Sigmoid()

        # Linear function 2: 84 --> 2
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)

        self.sigmoid = torch.nn.Sigmoid()

        self.fc3 = torch.nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)

        # Non-linearity 1
        out = self.sigmoid(out)

        # Linear function 2 (readout)
        out = self.fc2(out)

        out = self.sigmoid(out)

        out = self.fc3(out)
        return out

