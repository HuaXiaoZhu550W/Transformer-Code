"""
超参数文件
"""
import argparse
from utils import get_vocab, try_all_gpus

parser = argparse.ArgumentParser()
parser.add_argument('-max_len', type=int, default=128)
parser.add_argument('-embed_dim', type=int, default=512)
parser.add_argument('-ffn_hiddens', type=int, default=2048)
parser.add_argument('-num_layers', type=int, default=6)
parser.add_argument('-num_heads', type=int, default=8)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-beta1', type=float, default=0.9)
parser.add_argument('-beta2', type=float, default=0.98)
parser.add_argument('-eps', type=float, default=1e-9)
parser.add_argument('-k', type=int, default=2)
parser.add_argument('-batch_size', type=int, default=2)
parser.add_argument('-eval_batch', type=int, default=1)
parser.add_argument('-epochs', type=int, default=1000)
parser.add_argument('-warmup_steps', type=int, default=4000)
parser.add_argument('-device', type=str, default=try_all_gpus())  # 返回一个list,包含多GPU, 单CPU则需要device[0]
parser.add_argument('-train_path',
                    default={
                        'zh': "F:/code_space/NLP/Data/zh_en_translation/train/news-commentary-v13.zh-en.zh",
                        'en': "F:/code_space/NLP/Data/zh_en_translation/train/news-commentary-v13.zh-en.en"})
parser.add_argument('-test_path',
                    default={'zh': "F:/code_space/NLP/Data/zh_en_translation/test/newstest2017.tc.zh",
                             'en': "F:/code_space/NLP/Data/zh_en_translation/test/newstest2017.tc.en"})
parser.add_argument('-src_vocab', default=get_vocab('source.pkl'))
parser.add_argument('-tar_vocab', default=get_vocab('target.pkl'))
parser.add_argument('-weight_path', default="weight/")
parser.add_argument('-logs_path', default="logs/")
opt = parser.parse_args(args=[])
