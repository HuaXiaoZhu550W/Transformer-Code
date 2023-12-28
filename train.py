from tqdm import tqdm
import logging
from utils import grad_clipping


def train_epoch(model, dataloader, optimizer, loss_fn, lr_scheduler, device, epoch):
    """
    训练一个epoch
    """
    model.to(device)
    model.train()

    total_loss = 0.
    iterations = len(dataloader)

    # 创建进度条对象
    pbar = tqdm(desc=f'epoch: {epoch + 1}', total=iterations, postfix=dict, mininterval=0.4)

    for iteration, batch_iter in enumerate(dataloader):
        source, target, source_length, target_length = [x.to(device) for x in batch_iter]
        output = model(source, source_length, target[:, :-1], target_length)
        optimizer.zero_grad()
        loss = loss_fn(output, target[:, 1:], target_length)
        loss.backward()
        grad_clipping(model, 1)
        lr_scheduler.step()  # 更新学习率
        optimizer.step()

        # 记录训练日志
        logging.info(f"Epoch: {epoch + 1}, Batch: {iterations * epoch + iteration+1}, Loss: {loss.item()}")

        total_loss += loss.item()

        pbar.set_postfix(**{'loss': f"{total_loss / (iteration + 1):.4f}",
                            'lr': f"{optimizer.param_groups[0]['lr']:.6f}",
                            'device': f"{source.device}"})
        pbar.update(1)
    pbar.close()
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.module.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()
                  }
    return checkpoint
