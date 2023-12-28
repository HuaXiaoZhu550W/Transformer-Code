import torch.nn as nn
from utils import make_loss_mask


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """
    带遮蔽的softmax交叉熵损失函数
    pred的形状：(batch_size, max_length)
    target的形状：(batch_size, max_length)
    target_length的形状：(batch_size, )
    """
    def __init__(self, max_length, reduction='mean', **kwargs):
        super(MaskedSoftmaxCELoss, self).__init__(**kwargs)
        self.max_length = max_length
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, target, target_length):
        loss_mask = make_loss_mask(self.max_length, target_length)
        mask_loss = self.loss(pred.permute(0, 2, 1), target)*loss_mask.view(-1)
        loss = mask_loss.sum() / loss_mask.sum()
        return loss
