import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, dropout, **kwargs):
        super(TransformerEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim,
                                      padding_idx=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        # X shape:(N, max_len)
        token_embedding = self.embedding(X) * self.embed_dim ** 0.5
        positional_encoding = self.make_positional_encoding().to(X.device)
        return self.dropout(token_embedding + positional_encoding[:, :X.shape[1]])

    def make_positional_encoding(self):
        """生成位置编码"""
        # positions shape: (1, max_length, embed_dim)
        positions = torch.zeros((1, self.max_len, self.embed_dim))
        x = torch.arange(self.max_len).reshape((-1, 1)) / torch.pow(10000, torch.arange(0, self.embed_dim, 2) / self.embed_dim)
        positions[:, :, 0::2] = torch.sin(x)
        positions[:, :, 1::2] = torch.sin(x)
        return positions
