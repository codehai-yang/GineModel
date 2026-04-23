import os
import torch
import LoadSample as loadSample
import GINEClassifier as gineModel
import GlobalConfig as config
import train_and_evaluate as trainAndEval
from GraphDataset import GraphDataset
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_name')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ─────────────────────────────────────────────────────────────
#  DataLoader 参数
#  num_workers: 并行预加载进程数
#    - Windows 建议设为 0（多进程在 Windows 有坑）
#    - Linux/Mac 建议设为 4~8（按 CPU 核数调）
#  pin_memory: True 时 CPU→GPU 的数据传输更快（需要 CUDA）
#  persistent_workers: 避免每个 epoch 重建 worker 进程
# ─────────────────────────────────────────────────────────────
NUM_WORKERS_TRAIN = 2
NUM_WORKERS_EVAL  = 1
PIN_MEMORY        = device.type == 'cuda'


def build_loaders(file_list, train_indices, val_indices, test_indices):
    """
    用 GraphDataset + DataLoader 替换原来的手动切片循环。
    DataLoader 会：
        1. 用 NUM_WORKERS 个进程并行调用 dataset.get()（读文件+归一化）
        2. 自动把多个 Data 对象 collate 成一个 Batch（大图拼接）
        3. 把 Batch 放到 pin_memory 缓冲区等待 .to(device)
    """
    train_loader = DataLoader(
        GraphDataset(file_list, train_indices),
        batch_size       = config.BATCH_SIZE,
        shuffle          = True,            # 每 epoch 自动 shuffle
        num_workers      = NUM_WORKERS_TRAIN,
        pin_memory       = PIN_MEMORY,
        persistent_workers = NUM_WORKERS_TRAIN > 0,
        drop_last        = False,
    )
    val_loader = DataLoader(
        GraphDataset(file_list, val_indices),
        batch_size       = config.BATCH_SIZE,
        shuffle          = False,
        num_workers      = NUM_WORKERS_EVAL,
        pin_memory       = PIN_MEMORY,
        persistent_workers = NUM_WORKERS_EVAL > 0,
    )
    test_loader = DataLoader(
        GraphDataset(file_list, test_indices),
        batch_size       = config.BATCH_SIZE,
        shuffle          = False,
        num_workers      = NUM_WORKERS_EVAL,
        pin_memory       = PIN_MEMORY,
        persistent_workers = NUM_WORKERS_EVAL > 0,
    )
    return train_loader, val_loader, test_loader


def train(fileDir):
    """
    完整训练流程：
    1. 建立全局索引
    2. 划分训练/验证/测试集，构建 DataLoader
    3. 训练循环（含早停）
    4. 最终测试评估 + 保存 Excel
    """

    # ===== 第一步：建立全局索引并划分数据集 =====
    print('=' * 50)
    print('建立全局索引...')
    all_indices, file_list = loadSample.build_global_indices(fileDir)

    print('\n划分数据集...')
    train_indices, val_indices, test_indices = loadSample.split_indices(all_indices)

    print('\n构建 DataLoader...')
    train_loader, val_loader, test_loader = build_loaders(
        file_list, train_indices, val_indices, test_indices
    )
    print(
        f'  训练集: {len(train_indices)} 样本 | '
        f'验证集: {len(val_indices)} 样本 | '
        f'测试集: {len(test_indices)} 样本'
    )

    # ===== 第二步：初始化模型和优化器 =====
    print('\n初始化模型...')
    model     = gineModel.CostModelV2().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = 0.001,
        weight_decay = 0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = config.NUM_EPOCHS,
        eta_min= 1e-6
    )

    best_val_loss = float('inf')
    no_improve    = 0
    batch_count   = 0

    # ===== 第三步：训练循环 =====
    print('\n开始训练...')
    print('=' * 50)

    for epoch in range(config.NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}')

        epoch_train_loss = 0.0
        total_samples    = 0
        stop_training    = False

        # ── 直接迭代 DataLoader，每次得到一个已经打包好的 Batch ──
        for batch in train_loader:
            batch_loss_sum, batch_size = trainAndEval.train_one_batch(model, optimizer, batch, device)

            epoch_train_loss += batch_loss_sum
            total_samples    += batch_size
            batch_count      += 1

            # 每隔 VALIDATE_EVERY 个 batch 验证一次
            if batch_count % config.VALIDATE_EVERY == 0:
                val_loss   = trainAndEval.evaluate(
                    model, val_loader, device,
                    max_samples=config.VAL_BATCH_SIZE
                )
                current_lr = optimizer.param_groups[0]['lr']

                batch_loss_mean = batch_loss_sum / batch_size
                print(
                    f'  batch {batch_count:6d} | '
                    f'训练loss: {batch_loss_mean:.4f} | '
                    f'验证loss: {val_loss:.4f} | '
                    f'学习率: {current_lr:.6f}'
                )

                writer.add_scalar('Loss/trainBatch',   batch_loss_mean,  batch_count)
                writer.add_scalar('Loss/valBatch',     val_loss,    batch_count)
                writer.add_scalar('Learning Rate/batch', current_lr, batch_count)


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve    = 0
                    torch.save(model.state_dict(), config.MODEL_SAVE)
                    print(f'  → 验证loss改善至 {best_val_loss:.4f}，模型已保存')
                else:
                    no_improve += 1
                    print(f'  → 验证loss无改善，已连续 {no_improve}/{config.PATIENCE} 次')
                    if no_improve >= config.PATIENCE:
                        print(f'\n早停：连续 {config.PATIENCE} 次验证无改善，停止训练')
                        stop_training = True
                        break
        if stop_training:
            break

        avg_epoch_loss = epoch_train_loss / total_samples
        print(f'Epoch {epoch + 1} 平均训练loss: {avg_epoch_loss:.4f}')
        writer.add_scalar('Loss/trainEpochAvg', avg_epoch_loss, epoch)
        scheduler.step()

    writer.close()

    # ===== 第四步：最终测试评估 =====
    print('\n' + '=' * 50)
    print('训练结束，开始最终测试...')

    model.load_state_dict(torch.load(config.MODEL_SAVE, map_location=device))
    print(f'已加载最优模型（验证loss: {best_val_loss:.4f}）')

    test_loss = trainAndEval.evaluate(model, test_loader, device)
    print(f'最终测试loss: {test_loss:.4f}')
    print('=' * 50)

    # ===== 第五步：保存预测结果到 Excel =====
    print('\n保存预测结果到 Excel...')
    model_dir  = os.path.dirname(config.MODEL_SAVE)
    excel_path = os.path.join(model_dir, 'test_predictions.xlsx')
    trainAndEval.evaluate_and_save_results(
        model, test_loader, excel_path, device
    )
    return model


if __name__ == '__main__':
    train(config.SAMPLE_SAVE)