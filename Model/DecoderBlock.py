import torch.nn as nn
from .MultiHeadAttention import MultiHeadAttention
from .PositionWiseFFN import PositionWiseFFN
from .AddNorm import AddNorm


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hiddens, dropout, **kwargs):
        super(TransformerDecoderBlock, self).__init__(**kwargs)
        self.attention1 = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.addnorm1 = AddNorm(norm_shape=embed_dim, dropout=dropout)
        self.attention2 = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.addnorm2 = AddNorm(norm_shape=embed_dim, dropout=dropout)
        self.ffn = PositionWiseFFN(ffn_inputs=embed_dim, ffn_hiddens=ffn_hiddens, ffn_outputs=embed_dim)
        self.addnorm3 = AddNorm(norm_shape=embed_dim, dropout=dropout)

    def forward(self, X, encoder_out, memory_mask, target_mask):
        X_attention = self.attention1(keys=X, values=X, queries=X, mask=target_mask)
        queries = self.addnorm1(X, X_attention)
        attention_out = self.attention2(keys=encoder_out, values=encoder_out, queries=queries, mask=memory_mask)
        addnorm_out = self.addnorm2(queries, attention_out)
        return self.addnorm3(addnorm_out, self.ffn(addnorm_out))
