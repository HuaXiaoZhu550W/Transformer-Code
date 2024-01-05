import torch
from config import opt
from torch.utils.data import Dataset
from utils import tokenize


class TextDataset(Dataset):
    def __init__(self, source_text, target_text, max_length):
        super(TextDataset, self).__init__()
        self.source = tokenize(source_text)
        self.target = tokenize(target_text)
        self.source_vocab = opt.src_vocab
        self.target_vocab = opt.tar_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        if len(self.source[idx]) > self.max_length-2:
            self.source[idx] = self.source[idx][: self.max_length-2]
            source_length = self.max_length
        else:
            source_length = len(self.source[idx]) + 2

        if len(self.target[idx]) > self.max_length-1:
            self.target[idx] = self.target[idx][: self.max_length-1]
            target_length = self.max_length
        else:
            target_length = len(self.target[idx]) + 1

        source = torch.tensor([self.source_vocab['<bos>']] +
                              self.source_vocab[self.source[idx]] +
                              [self.source_vocab['<eos>']] +
                              [self.source_vocab['<pad>']] * (self.max_length-source_length))
        target = torch.tensor(self.target_vocab[self.target[idx]] +
                              [self.target_vocab['<eos>']] +
                              [self.target_vocab['<pad>']] * (self.max_length - target_length))

        return source, target, source_length, target_length
