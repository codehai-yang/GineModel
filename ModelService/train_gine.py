import os
import torch
import LoadSample as loadSample
import GINEClassifier as gineModel
import argparse
import GlobalConfig as config
import train_and_evaluate as trainAndEval
from GraphDataset import GraphDataset
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter


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

def build_loaders(file_list, train_indices, val_indices, test_indices,
                  batch_size, num_workers_train, num_workers_eval, pin_memory):
    """
    用 GraphDataset + DataLoader 替换原来的手动切片循环。
    DataLoader 会：
        1. 用 NUM_WORKERS 个进程并行调用 dataset.get()（读文件+归一化）
        2. 自动把多个 Data 对象 collate 成一个 Batch（大图拼接）
        3. 把 Batch 放到 pin_memory 缓冲区等待 .to(device)
    """
    train_loader = DataLoader(
        GraphDataset(file_list, train_indices),
        batch_size       = batch_size,
        shuffle          = True,
        num_workers      = num_workers_train,
        pin_memory       = pin_memory,
        persistent_workers = num_workers_train > 0,
        drop_last        = False,
    )
    val_loader = DataLoader(
        GraphDataset(file_list, val_indices),
        batch_size       = batch_size,
        shuffle          = False,
        num_workers      = num_workers_eval,
        pin_memory       = pin_memory,
        persistent_workers = num_workers_eval > 0,
    )
    test_loader = DataLoader(
        GraphDataset(file_list, test_indices),
        batch_size       = batch_size,
        shuffle          = False,
        num_workers      = num_workers_eval,
        pin_memory       = pin_memory,
        persistent_workers = num_workers_eval > 0,
    )
    return train_loader, val_loader, test_loader


def train(args):
    """
    完整训练流程：
    1. 建立全局索引
    2. 划分训练/验证/测试集，构建 DataLoader
    3. 训练循环（含早停）
    4. 最终测试评估 + 保存 Excel
    """

    # ===== 配置参数（命令行优先，否则使用默认配置）=====
    data_dir = args.data_dir if args.data_dir else config.SAMPLE_SAVE
    model_save = args.model_save if args.model_save else config.MODEL_SAVE
    batch_size = args.batch_size if args.batch_size else config.BATCH_SIZE
    num_epochs = args.epochs if args.epochs else config.NUM_EPOCHS
    learning_rate = args.lr if args.lr else config.LEARNING_RATE
    weight_decay = args.weight_decay if args.weight_decay is not None else 0.01
    patience = args.patience if args.patience else config.PATIENCE
    validate_every = args.validate_every if args.validate_every else config.VALIDATE_EVERY
    val_batch_size = args.val_batch_size if args.val_batch_size else config.VAL_BATCH_SIZE
    hidden_dim = args.hidden_dim if args.hidden_dim else config.HIDDEN_DIM
    num_layers = args.num_layers if args.num_layers else config.NUM_LAYERS
    num_workers_train = args.num_workers_train if args.num_workers_train is not None else 2
    num_workers_eval = args.num_workers_eval if args.num_workers_eval is not None else 1
    seed = args.seed if args.seed else config.RANDOM_SEED

    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 初始化 TensorBoard
    # 初始化设备和 TensorBoard（只在主进程执行）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin_memory = device.type == 'cuda'
    writer = SummaryWriter(args.log_dir)


    print('=' * 50)
    print('训练配置:')
    print(f'  数据目录: {data_dir}')
    print(f'  模型保存: {model_save}')
    print(f'  设备: {device}')
    print(f'  批次大小: {batch_size}')
    print(f'  训练轮数: {num_epochs}')
    print(f'  学习率: {learning_rate}')
    print(f'  权重衰减: {weight_decay}')
    print(f'  隐藏层维度: {hidden_dim}')
    print(f'  GINE层数: {num_layers}')
    print(f'  早停耐心值: {patience}')
    print(f'  验证频率: {validate_every}')
    print(f'  验证采样数: {val_batch_size}')
    print(f'  Worker数(训练/验证): {num_workers_train}/{num_workers_eval}')
    print(f'  随机种子: {seed}')
    print('=' * 50)

    # ===== 第一步：建立全局索引并划分数据集 =====
    print('\n建立全局索引...')
    all_indices, file_list = loadSample.build_global_indices(data_dir)

    print('\n划分数据集...')
    train_indices, val_indices, test_indices = loadSample.split_indices(all_indices)

    print('\n构建 DataLoader...')
    train_loader, val_loader, test_loader = build_loaders(
        file_list, train_indices, val_indices, test_indices,
        batch_size, num_workers_train, num_workers_eval, pin_memory
    )
    print(
        f'  训练集: {len(train_indices)} 样本 | '
        f'验证集: {len(val_indices)} 样本 | '
        f'测试集: {len(test_indices)} 样本'
    )

    # ===== 第二步：初始化模型和优化器 =====
    print('\n初始化模型...')
    model = gineModel.CostModelV2(
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    no_improve = 0
    batch_count = 0

    # ===== 第三步：训练循环 =====
    print('\n开始训练...')
    print('=' * 50)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        epoch_train_loss = 0.0
        total_samples = 0
        stop_training = False

        # ── 直接迭代 DataLoader，每次得到一个已经打包好的 Batch ──
        for batch in train_loader:
            batch_loss_sum, batch_size_actual = trainAndEval.train_one_batch(model, optimizer, batch, device)

            epoch_train_loss += batch_loss_sum
            total_samples += batch_size_actual
            batch_count += 1

            # 每隔 VALIDATE_EVERY 个 batch 验证一次
            if batch_count % validate_every == 0:
                val_loss = trainAndEval.evaluate(
                    model, val_loader, device,
                    max_samples=val_batch_size
                )
                current_lr = optimizer.param_groups[0]['lr']

                batch_loss_mean = batch_loss_sum / batch_size_actual
                print(
                    f'  batch {batch_count:6d} | '
                    f'训练loss: {batch_loss_mean:.4f} | '
                    f'验证loss: {val_loss:.4f} | '
                    f'学习率: {current_lr:.6f}'
                )

                writer.add_scalar('Loss/trainBatch', batch_loss_mean, batch_count)
                writer.add_scalar('Loss/valBatch', val_loss, batch_count)
                writer.add_scalar('Learning Rate/batch', current_lr, batch_count)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    # 确保目录存在
                    os.makedirs(os.path.dirname(model_save), exist_ok=True)
                    torch.save(model.state_dict(), model_save)
                    print(f'  → 验证loss改善至 {best_val_loss:.4f}，模型已保存')
                else:
                    no_improve += 1
                    print(f'  → 验证loss无改善，已连续 {no_improve}/{patience} 次')
                    if no_improve >= patience:
                        print(f'\n早停：连续 {patience} 次验证无改善，停止训练')
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

    model.load_state_dict(torch.load(model_save, map_location=device))
    print(f'已加载最优模型（验证loss: {best_val_loss:.4f}）')

    test_loss = trainAndEval.evaluate(model, test_loader, device)
    print(f'最终测试loss: {test_loss:.4f}')
    print('=' * 50)

    # ===== 第五步：保存预测结果到 Excel =====
    print('\n保存预测结果到 Excel...')
    model_dir = os.path.dirname(model_save)
    excel_path = os.path.join(model_dir, 'test_predictions.xlsx')
    trainAndEval.evaluate_and_save_results(
        model, test_loader, excel_path, device
    )
    return model

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='GINE 模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ===== 必需参数 =====
    parser.add_argument('--data_dir', type=str, default=None,
                        help='样本数据目录路径（默认使用 GlobalConfig.SAMPLE_SAVE）')
    parser.add_argument('--model_save', type=str, default=None,
                        help='模型保存路径（默认使用 GlobalConfig.MODEL_SAVE）')

    # ===== 训练超参数 =====
    parser.add_argument('--batch_size', type=int, default=None,
                        help=f'批次大小（默认: {config.BATCH_SIZE}）')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'最大训练轮数（默认: {config.NUM_EPOCHS}）')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'学习率（默认: {config.LEARNING_RATE}）')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help=f'权重衰减系数（默认: 0.01）')
    parser.add_argument('--patience', type=int, default=None,
                        help=f'早停耐心值（默认: {config.PATIENCE}）')

    # ===== 模型架构参数 =====
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help=f'GINE隐藏层维度（默认: {config.HIDDEN_DIM}）')
    parser.add_argument('--num_layers', type=int, default=None,
                        help=f'GINE层数（默认: {config.NUM_LAYERS}）')

    # ===== 验证参数 =====
    parser.add_argument('--validate_every', type=int, default=None,
                        help=f'每隔多少个batch验证一次（默认: {config.VALIDATE_EVERY}）')
    parser.add_argument('--val_batch_size', type=int, default=None,
                        help=f'每次验证采样数量（默认: {config.VAL_BATCH_SIZE}）')

    # ===== 数据加载参数 =====
    parser.add_argument('--num_workers_train', type=int, default=None,
                        help=f'训练集数据加载worker数（默认: 2）')
    parser.add_argument('--num_workers_eval', type=int, default=None,
                        help=f'验证/测试集数据加载worker数（默认: 1）')

    # ===== 其他参数 =====
    parser.add_argument('--seed', type=int, default=None,
                        help=f'随机种子（默认: {config.RANDOM_SEED}）')
    parser.add_argument('--log_dir', type=str, default='runs/experiment_name',
                        help='TensorBoard日志目录（默认: runs/experiment_name）')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    train(args)