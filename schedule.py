import torch


# 学习率下降策略(热身反平方衰减)
class WarmupInverseSqrtDecay:
    def __init__(self, optimizer, d_model, warmup_steps):
        super(WarmupInverseSqrtDecay, self).__init__()
        self.optimizer = optimizer
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.steps = 1.0  # 当前step

    def step(self):
        lr = self.optimizer.param_groups[0]['lr'] * (self.d_model ** -0.5) * \
             min(self.steps ** -0.5, self.steps * (self.warmup_steps ** -1.5))
        self.steps += 1.
        # 更新学习率
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr
