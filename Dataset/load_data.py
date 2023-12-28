from torch.utils.data import DataLoader
from utils import read_data
from .MyDataset import TextDataset
from config import opt


def load_data(zh_path, en_path, batch_size, is_train=True, num_workers=1):
    zh_text = read_data(zh_path)
    en_text = read_data(en_path)
    dataset = TextDataset(source_text=zh_text, target_text=en_text, max_length=opt.max_len)
    dataloader = DataLoader(dataset, batch_size, shuffle=is_train, num_workers=num_workers, drop_last=True)
    return dataloader
