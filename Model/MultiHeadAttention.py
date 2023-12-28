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

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_layer = nn.Linear(self.head_dim * num_heads, embed_dim)

    def forward(self, keys, values, queries, mask):
        N = queries.shape[0]  # batch_size
        key_len, value_len, query_len = keys.shape[1], values.shape[1], queries.shape[1]

        # 将embed_dim 拆分为num_heads份, 每一份head_dim, reshape后放入Linear层
        # keys shape: (N, key_len, embed_dim) reshape: (N, key_len, num_heads, head_dim)
        # values shape: (N, value_len, embed_dim) reshape: (N, value_len, num_heads, head_dim)
        # queries shape: (N, query_len, embed_dim) reshape: (N, query_len, num_heads, head_dim)

        keys = self.keys(keys.reshape(N, self.num_heads, key_len, self.head_dim))
        values = self.values(values.reshape(N, self.num_heads, value_len, self.head_dim))
        queries = self.queries(queries.reshape(N, self.num_heads, query_len, self.head_dim))

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
        # attention*values -> att_out shape: (N, query_len, num_heads, head_dim)
        att_out = torch.einsum('nhql, nhld -> nhqd', [attention, values])

        output = self.out_layer(att_out.reshape(N, query_len, self.num_heads * self.head_dim))

        return output
