import torch
import torch.nn as nn
import torch.nn.functional as F


class NetModel(nn.Module):
    def __init__(self, width=64):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(16, width),
            nn.LeakyReLU(0.2),
            nn.Linear(width, 4),
        )

        nn.init.xavier_normal_(self.layers[0].weight.data)
        nn.init.xavier_normal_(self.layers[2].weight.data)
        nn.init.zeros_(self.layers[0].bias.data)
        nn.init.zeros_(self.layers[2].bias.data)

    def forward(self, x):
        x = x.view(-1, 16).to(torch.float32)
        return self.layers(x)

    def get_prob(self, x):
        with torch.no_grad():
            prob = F.softmax(self.forward(x), dim=-1).view(-1)
            prob = prob.to(torch.float64)
            return prob/prob.sum()
