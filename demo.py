import os
import torch
import jieba
from utils import bleu
from config import opt
from Model import Transformer


def translation(source):
    # 搭建模型
    model = Transformer(source_vocab=opt.src_vocab,
                        target_vocab=opt.tar_vocab,
                        embed_dim=opt.embed_dim,
                        num_heads=opt.num_heads,
                        ffn_hiddens=opt.ffn_hiddens,
                        num_layers=opt.num_layers,
                        max_len=opt.max_len,
                        dropout=opt.dropout
                        )
    # 加载模型参数
    checkpoint = torch.load(os.path.join(opt.weight_path, 'checkpoint.pth'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model.to('cpu')

    # 数据预处理
    sentence_token = [' '.join(jieba.cut(s + '。')).split(' ') for s in source.split("。")[: -1]]
    translates = []
    for st in sentence_token:
        if len(st) > opt.max_len - 2:
            st = st[: opt.max_len - 2]
            source_length = opt.max_len
        else:
            source_length = len(st) + 2
        src_tensor = torch.tensor([[opt.src_vocab['<bos>']] + opt.src_vocab[st] +
                                   [opt.src_vocab['<eos>']] +
                                   [opt.src_vocab['<pad>']] * (opt.max_len - source_length)])
        source_length = torch.tensor([source_length])
        outputs = [opt.tar_vocab['<bos>']]  # decoder初始输入

        for i in range(opt.max_len):
            dec_input = torch.tensor(outputs).unsqueeze(0)
            dec_len = torch.tensor([len(outputs)])  # 输入长度
            with torch.no_grad():
                output = model(src_tensor, source_length, dec_input, dec_len)
            pred = torch.argmax(output, dim=-1)[:, -1]
            if pred.item() == opt.tar_vocab['<eos>']:
                break
            outputs.append(pred.item())

        translate = ' '.join(opt.tar_vocab.to_tokens(outputs[1:]))
        translates.append(translate)
    return ' '.join(translates)


if __name__ == "__main__":
    inputs = "他是一个记者。"
    labels = "he is a reporter."
    preds = translation(inputs)
    sentence_bleu = bleu(preds, labels, opt.k)
    print(f"source: {inputs}\n -> \ntranslate: {preds} \nbleu: {sentence_bleu:.4f}")
