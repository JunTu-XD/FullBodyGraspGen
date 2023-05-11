from torch import nn


class Eps(nn.Module):
    def __init__(self, D):
        super.__init__()
        self.model = nn.Sequential(*[nn.Linear(D, D)])

    def forward(self, x, condition, t):
        ## x + t encoding
        return self.model(x)
