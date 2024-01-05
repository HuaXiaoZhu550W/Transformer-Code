import pickle
from utils import read_data, tokenize, Vocab
# source_path = "F:/code_space/NLP/Data/zh_en_translation/train/news-commentary-v13.zh-en.zh"
# target_path = "F:/code_space/NLP/Data/zh_en_translation/train/news-commentary-v13.zh-en.en"
source_path = "F:/code_space/NLP/Data/zh_en_translation/dev/newsdev2017.tc.zh"
target_path = "F:/code_space/NLP/Data/zh_en_translation/dev/newsdev2017.tc.en"


# 生成train vocab
zh_text = read_data(source_path)
en_text = read_data(target_path)

zh_tokenize = tokenize(zh_text)
en_tokenize = tokenize(en_text)

# 生成源语言词表
src_vocab = Vocab(tokens=zh_tokenize, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
with open('source.pkl', 'wb') as f:
    pickle.dump(src_vocab, f)

# 生成目标语言词表
tar_vocab = Vocab(tokens=en_tokenize, min_freq=1, reserved_tokens=['<pad>', '<bos>', '<eos>'])
with open('target.pkl', 'wb') as f:
    pickle.dump(tar_vocab, f)


