# 学习率下降策略(热身反平方衰减)
class WarmupInverseSqrtDecay:
    def __init__(self, optimizer, d_model, warmup_steps):
        """
        初始化学习率调整器。
        :param optimizer: 优化器对象
        :param d_model: 模型的维度
        :param warmup_steps: 热身步数
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0  # 当前步数

    def step(self):
        """
        更新学习率并增加步数。
        :return: 当前学习率
        """
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
