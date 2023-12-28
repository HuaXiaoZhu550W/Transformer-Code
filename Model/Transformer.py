from utils import *
from .EncoderBlock import TransformerEncoderBlock
from .DecoderBlock import TransformerDecoderBlock


# 搭建模型

class TransformerEncoder(nn.Module):
    def __init__(self, source_vocab_size, embed_dim, num_heads, ffn_hiddens, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(num_embeddings=source_vocab_size, embedding_dim=embed_dim)
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(name=f'TransformerEncoderBlock{i}',
                                   module=TransformerEncoderBlock(embed_dim, num_heads, ffn_hiddens, dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask):
        positions = make_positional_encoding(self.embed_dim, X.shape[1]).to(X.device)
        # 因为位置编码值在-1和1之间, 因此嵌入值乘以嵌入维度的平方根进行缩放, 然后再与位置编码相加
        output = self.dropout(self.word_embedding(X) * math.sqrt(self.embed_dim) +
                              positions[:, :self.word_embedding(X).shape[-2]])
        for layer in self.layers:
            output = layer(output, output, output, mask)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, num_heads, ffn_hiddens, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_embeddings=target_vocab_size, embedding_dim=embed_dim)
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.add_module(name=f'TransformerDecoderBlock{i}',
                                   module=TransformerDecoderBlock(embed_dim, num_heads, ffn_hiddens, dropout))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=embed_dim, out_features=target_vocab_size)

    def forward(self, X, encoder_out, memory_mask, target_mask):
        positions = make_positional_encoding(self.embed_dim, X.shape[1]).to(X.device)
        # 因为位置编码值在-1和1之间, 因此嵌入值乘以嵌入维度的平方根进行缩放, 然后再与位置编码相加
        output = self.dropout(self.word_embedding(X) * math.sqrt(self.embed_dim) +
                              positions[:, :self.word_embedding(X).shape[-2]])
        for layer in self.layers:
            output = layer(output, encoder_out, memory_mask, target_mask)
        return self.fc(output)


class Transformer(nn.Module):
    def __init__(
            self,
            source_vocab,
            target_vocab,
            embed_dim=512,
            num_heads=8,
            ffn_hiddens=2048,
            num_layers=6,
            dropout=0.1):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.encoder = TransformerEncoder(len(source_vocab), embed_dim, num_heads, ffn_hiddens,
                                          num_layers, dropout)
        self.decoder = TransformerDecoder(len(target_vocab), embed_dim, num_heads, ffn_hiddens,
                                          num_layers, dropout)

    def forward(self, source, source_lens, target, target_lens):
        source_mask = make_pad_mask(source_lens, source.shape[1])
        memory_mask = make_memory_mask(source_lens, target_lens, source.shape[1], target.shape[1])
        target_mask = make_target_mask(target_lens, target.shape[1])
        encoder_out = self.encoder(source, source_mask)
        output = self.decoder(target, encoder_out, memory_mask, target_mask)
        return output
