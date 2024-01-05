import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from Dataset import load_data
from Model import Transformer
from loss import MaskedSoftmaxCELoss
from schedule import WarmupInverseSqrtDecay
from utils import init_weights
from train import train_epoch
from eval import evaluate
from config import opt


def main(is_continue):
    # 加载数据
    train_dataloader = load_data(zh_path=opt.train_path['zh'], en_path=opt.train_path['en'],
                                 batch_size=opt.batch_size, is_train=True, num_workers=4)

    valid_dataloader = load_data(zh_path=opt.test_path['zh'], en_path=opt.test_path['en'],
                                 batch_size=opt.eval_batch, is_train=False, num_workers=4)

    # 加载模型
    model = Transformer(source_vocab=opt.src_vocab,
                        target_vocab=opt.tar_vocab,
                        embed_dim=opt.embed_dim,
                        num_heads=opt.num_heads,
                        ffn_hiddens=opt.ffn_hiddens,
                        num_layers=opt.num_layers,
                        max_len=opt.max_len,
                        dropout=opt.dropout
                        )

    # 在多个GPU上设置模型
    model = nn.DataParallel(model, device_ids=opt.device)

    # 损失函数
    loss_fn = MaskedSoftmaxCELoss(max_length=opt.max_len)

    # 参数初始化
    if is_continue:
        # 加载已保存的模型参数
        checkpoint = torch.load(os.path.join(opt.weight_path, 'checkpoint.pth'))
        epoch = checkpoint['epoch'] + 1
        steps = checkpoint['steps']
        lr = checkpoint['lr']
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        epoch = 0
        steps = 0
        lr = opt.lr
        model.apply(init_weights)

    # 优化器
    optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    # 学习率衰减策略
    lr_scheduler = WarmupInverseSqrtDecay(optimizer, d_model=opt.embed_dim, warmup_steps=opt.warmup_steps, steps=steps)

    # 配置日志记录器
    if not os.path.exists(opt.logs_path):
        os.mkdir(opt.logs_path)
    logging.basicConfig(filename=os.path.join(opt.logs_path, 'train.log'), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 开始训练
    for epoch in range(epoch, epoch + opt.epochs):
        checkpoint = train_epoch(model, train_dataloader, optimizer, loss_fn, lr_scheduler, opt.device[0], epoch)
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(opt.weight_path, 'checkpoint.pth'))
    # 模型评估
    mean_bleu = evaluate(model.module, valid_dataloader, opt.device[0])
    print(f"bleu: {mean_bleu:.4f}")


if __name__ == "__main__":
    main(is_continue=False)
