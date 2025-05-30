import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedBlockTorch(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.ln(x)
        x = self.linear(x)
        x = F.gelu(x)
        return x
