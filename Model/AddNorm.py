import torch.nn as nn


class AddNorm(nn.Module):
    """残差链接和层归一化"""
    def __init__(self, norm_shape, dropout=0.1, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape=norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
