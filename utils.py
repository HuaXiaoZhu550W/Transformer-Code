import torch
import math
import pickle
import torch.nn as nn
import collections


def read_data(data_path):
    """读取数据"""
    with open(data_path, 'r', encoding='utf8') as f:
        return f.read()


def tokenize(text):
    """词元化"""
    tokens = []
    for line in text.split('\n'):
        tokens.append(line.split(' '))
    return tokens


def count_corpus(tokens):
    """统计词元的频率, tokens 是1D或者2D的list"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 如果tokens是空的或者是2D列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词汇表"""

    def __init__(self, tokens, min_freq, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)  # 统计词频
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # 按照词频排序
        self.idx_to_token = ['<unk>'] + reserved_tokens  # 未知词元的索引为0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """返回token对应的index"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]  # 递归返回tokens中每一个token的index

    def to_tokens(self, indexes):
        """返回index对应的token"""
        if not isinstance(indexes, (list, tuple)):
            return self.idx_to_token[indexes]
        return [self.idx_to_token[index] for index in indexes]

    @property
    def unk(self):
        return 0

    @property
    def _token_freqs(self):
        return self.token_freqs


def get_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def try_all_gpus():
    device = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return device if device else [torch.device('cpu')]


def make_positional_encoding(embed_dim, max_length):
    """生成位置编码"""
    # positions shape: (1, max_length, embed_dim)
    positions = torch.zeros((1, max_length, embed_dim))
    x = torch.arange(max_length).reshape((-1, 1)) / torch.pow(10000, torch.arange(0, embed_dim, 2) / embed_dim)
    positions[:, :, 0::2] = torch.sin(x)
    positions[:, :, 1::2] = torch.sin(x)
    return positions


def make_pad_mask(lens, max_length):
    """
    生成填充掩码
    lens: shape (batch_size,)
    max_length: int
    pad_mask: shape (batch_size, 1, source_max_length, source_max_length)
    """
    pad = torch.arange(max_length, device=lens.device)[None, :] < lens[:, None]
    pad = pad.unsqueeze(2).to(torch.float32)
    pad_mask = torch.bmm(pad, pad.transpose(1, 2))
    return pad_mask.unsqueeze(1)


def make_memory_mask(source_lens, target_lens, source_max_length, target_max_length):
    """
    生成encoder到decoder的掩码
    source_lens: shape (batch_size,)
    target_lens: shape (batch_size,)
    source_max_length: int
    target_max_length: int
    memory_mask: shape (batch_size, 1, target_max_length, source_max_length)
    """
    target = torch.arange(target_max_length, device=target_lens.device)[None, :] < target_lens[:, None]
    target = target.unsqueeze(2).to(torch.float32)
    source = torch.arange(source_max_length, device=source_lens.device)[None, :] < source_lens[:, None]
    source = source.unsqueeze(1).to(torch.float32)

    memory_mask = torch.bmm(target, source).unsqueeze(1)
    return memory_mask


def make_target_mask(target_lens, target_max_length):
    """
    生成decoder的输入掩码
    target_lens: shape (batch_size,)
    target_max_length: int
    target_mask: shape (batch_size, 1, target_max_length, target_max_length), type: torch.bool
    """
    pad_mask = make_pad_mask(target_lens, target_max_length)
    target_mask = torch.tril(torch.ones((target_max_length, target_max_length))).expand(
        (pad_mask.shape[0], 1, target_max_length, target_max_length)).to(target_lens.device)
    return target_mask.to(torch.bool) & pad_mask.to(torch.bool)


def make_loss_mask(max_length, target_length):
    """ 用于计算loss"""
    pad = torch.arange(max_length, device=target_length.device)[None, :] < target_length[:, None]
    pad = pad.to(torch.int64)
    return pad


def grad_clipping(net, theta):
    """梯度剪裁"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params if p.grad is not None))
    if norm > theta:
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens = pred_seq.split(' ')
    label_tokens = label_seq.split(' ')
    len_pred = len(pred_tokens)
    len_label = len(label_tokens)
    score = math.exp(min(0., 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches = 0
        label_subs = collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1 + 1e-5), math.pow(0.5, n))
    return score


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
