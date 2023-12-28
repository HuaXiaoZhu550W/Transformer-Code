import torch.nn as nn
from .MultiHeadAttention import MultiHeadAttention
from .PositionWiseFFN import PositionWiseFFN
from .AddNorm import AddNorm


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hiddens, dropout, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.addnorm1 = AddNorm(norm_shape=embed_dim, dropout=dropout)
        self.ffn = PositionWiseFFN(ffn_inputs=embed_dim, ffn_hiddens=ffn_hiddens, ffn_outputs=embed_dim)
        self.addnorm2 = AddNorm(norm_shape=embed_dim, dropout=dropout)

    def forward(self, keys, values, queries, mask):
        attention = self.attention(keys, values, queries, mask)
        X = self.addnorm1(queries, attention)
        return self.addnorm2(X, self.ffn(X))
