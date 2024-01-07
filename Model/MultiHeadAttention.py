import torch
import torch.nn as nn


# 多头自注意力机制

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads == embed_dim), "Embedding dim needs to be divisible by heads"

        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, keys, values, queries, mask):
        # 将embed_dim 拆分为num_heads份, 每一份head_dim
        # keys shape: (N, key_len, embed_dim) split后: (N, num_heads, key_len, head_dim)
        # values shape: (N, value_len, embed_dim) split后: (N, num_heads, value_len, head_dim)
        # queries shape: (N, query_len, embed_dim) split后: (N, num_heads, query_len head_dim)

        keys = self.split(self.keys(keys))
        values = self.split(self.values(values))
        queries = self.split(self.queries(queries))

        # queries*keys -> energy shape: (N, num_heads, query_len, key_len)
        energy = torch.einsum('nhqd, nhkd -> nhqk', [queries, keys]) / self.embed_dim ** (1 / 2)

        # 添加mask, shape: (N, 1, query_len, key_len)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        # 计算attention, 论文中除根号d的操作
        attention = torch.softmax(energy, dim=-1)

        # attention shape: (N, num_heads, query_len, key_len)
        # values shape: (N, num_heads, value_len, head_dim)
        # value_len == key_len, 都用l表示
        # attention*values -> att_out shape: (N, num_heads,query_len, head_dim)
        att_out = torch.einsum('nhql, nhld -> nhqd', [attention, values])

        return self.out_layer(self.concat(att_out))

    def split(self, X):
        # X shape: (N, max_len, embed_dim)
        X = X.view(X.shape[0], X.shape[1], self.num_heads, self.head_dim)
        return X.transpose(1, 2)

    def concat(self, X):
        # X shape: (N, num_heads, max_len, head_dim)
        X = X.transpose(1, 2)
        return X.reshape(X.shape[0], X.shape[1], self.embed_dim)
