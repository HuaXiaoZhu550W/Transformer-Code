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
                        dropout=opt.dropout
                        )
    # 加载模型参数
    checkpoint = torch.load(os.path.join(opt.weight_path, 'checkpoint.pth'))
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
                                   [opt.src_vocab['<pad>']] * (opt.max_len - source_length)]).to('cpu')
        source_length = torch.tensor([source_length])
        outputs = [opt.tar_vocab['<bos>']]  # decoder初始输入

        for i in range(opt.max_len):
            dec_input = torch.tensor(outputs).unsqueeze(0).to('cpu')
            dec_len = torch.tensor([len(outputs)]).to('cpu')  # 输入长度
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
    inputs = "在加利福尼亚州一个主要水务管理区披露州长杰瑞·布朗领导的行政当局将提供政府资金以完成两条巨型输水隧道的规划之后，有一些评论家和一位州议员表示，他们想进一步了解由谁来为州长所支持的拟耗资160亿美元的水务工程承担费用。"
    labels = "critics and a state lawmaker say they want more explanations on who &apos;s paying for a proposed $ 16 billion water project backed by Gov. Jerry Brown , after a leading California water district said Brown &apos;s administration was offering government funding to finish the planning for the two giant water tunnels ."
    preds = translation(inputs)
    sentence_bleu = bleu(preds, labels, opt.k)
    print(f"source: {inputs}\n -> \ntranslate: {preds} \nbleu: {sentence_bleu:.4f}")
