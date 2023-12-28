import torch.nn as nn


class PositionWiseFFN(nn.Module):
    """
    前馈神经网络, 包括两个线性层和一个ReLu激活函数, 输入输出为512, 中间维度为2048
    线性层可以使用两个1x1的卷积层替换
    """
    def __init__(self, ffn_inputs, ffn_hiddens, ffn_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(in_features=ffn_inputs, out_features=ffn_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=ffn_hiddens, out_features=ffn_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


# class PositionWiseFFN(nn.Module):
#     def __init__(self, ffn_inputs, ffn_hiddens, ffn_outputs, **kwargs):
#         super(PositionWiseFFN, self).__init__(**kwargs)
#         self.dense1 = nn.Conv1d(in_channels=ffn_inputs, out_channels=ffn_hiddens, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.dense2 = nn.Conv1d(in_channels=ffn_hiddens, out_channels=ffn_outputs, kernel_size=1)
#
#     def forward(self, X):
#         dense1_out = self.dense1(X.permute(0, 2, 1))
#         relu_out = self.relu(dense1_out)
#         output = self.dense2(relu_out)
#         return output.permute(0, 2, 1)
