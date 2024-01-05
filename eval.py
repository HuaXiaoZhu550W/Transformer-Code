import torch
from torch.nn.functional import cross_entropy
from tqdm import tqdm
from utils import bleu
from config import opt


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    iterations = len(dataloader)
    total_bleu = 0.0
    total_loss = 0.0

    # 创建进度条对象
    pbar = tqdm(desc=f'eval', total=iterations, postfix=dict, mininterval=0.4)
    for iteration, batch in enumerate(dataloader):
        loss = 0.0
        source, target, source_length, target_length = [x.to(device) for x in batch]
        outputs = [opt.tar_vocab['<bos>']]  # decoder初始输入
        for i in range(opt.max_len-1):
            dec_input = torch.tensor(outputs).unsqueeze(0).to(device)
            dec_len = torch.tensor([len(outputs)]).to(device)  # 输入长度
            with torch.no_grad():
                output = model(source, source_length, dec_input, dec_len)
                loss += cross_entropy(output[:, -1], target[:, i + 1], reduction='mean').item()
            pred = torch.argmax(output, dim=-1)[:, -1]
            if pred.item() == opt.tar_vocab['<eos>']:
                break

            outputs.append(pred.item())
        # 计算bleu
        translate = ' '.join(opt.tar_vocab.to_tokens(outputs[1:]))
        label = ' '.join(opt.tar_vocab.to_tokens(target[0][1: target_length.item()-1].to('cpu').numpy().tolist()))
        total_bleu += bleu(translate, label, opt.k)
        # 计算loss,每个词汇的平均loss
        total_loss += loss/len(outputs)
        pbar.set_postfix(**{'loss': f"{total_loss / (iteration + 1):.4f}"})
        pbar.update(1)
    pbar.close()

    return total_bleu / iterations


# if __name__ == "__main__":
#     from Model import Transformer
#     import os
#     from Dataset import load_data
#     # 搭建模型
#     model = Transformer(source_vocab=opt.src_vocab,
#                         target_vocab=opt.tar_vocab,
#                         embed_dim=opt.embed_dim,
#                         num_heads=opt.num_heads,
#                         ffn_hiddens=opt.ffn_hiddens,
#                         num_layers=opt.num_layers,
#                         max_len = opt.max_len,
#                         dropout=opt.dropout
#                         )
#     # 加载模型参数
#     checkpoint = torch.load(os.path.join(opt.weight_path, 'checkpoint.pth'), map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['model_state_dict'])
#
#     # 加载数据
#     valid_dataloader = load_data(zh_path=opt.test_path['zh'], en_path=opt.test_path['en'],
#                                  batch_size=opt.eval_batch, is_train=False, num_workers=4)
#     mean_bleu = evaluate(model, valid_dataloader, 'cpu')
#     print(f"bleu: {mean_bleu:.4f}")
