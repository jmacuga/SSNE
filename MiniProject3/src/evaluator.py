import torch
import torch.nn as nn


class Evaluator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Evaluator, self).__init__()
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc_out = nn.Linear(50, 43)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def get_features(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.bn1(x)
        x = self.LeakyReLU(x)
        x = self.fc_2(x)
        x = self.bn2(x)
        x = self.LeakyReLU(x)
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.fc_out(x)
        return x
