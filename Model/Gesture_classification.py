import torch
import torch.nn as nn


class GestureFNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(GestureFNN, self).__init__()
        # Linear function 1: 126 --> 100
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim_1)

        self.sigmoid = torch.nn.Sigmoid()

        # Linear function 2: 100 --> 64
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)

        self.sigmoid = torch.nn.Sigmoid()

        # Linear function 3: 64 --> 3  (output)
        self.fc3 = torch.nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        out = self.fc1(x)

        out = self.sigmoid(out)

        out = self.fc2(out)

        out = self.sigmoid(out)

        out = self.fc3(out)
        return out

