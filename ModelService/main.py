import random
import torch
import LoadSample as loadSample
import GINEClassifier as gineModel
import GlobalConfig as config
import train_and_evaluate as trainAndEval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

def train(fileDir):
    """
    完整训练流程：
    1. 建立全局索引
    2. 划分训练/验证/测试集
    3. 训练循环（含早停）
    4. 最终测试评估

    参数：
        file_list:       数据文件路径列表
    """

    # ===== 第一步：建立全局索引并划分数据集 =====
    print('=' * 50)
    print('建立全局索引...')
    all_indices,file_list = loadSample.build_global_indices(fileDir)

    print('\n划分数据集...')
    train_indices, val_indices, test_indices = loadSample.split_indices(all_indices)

    # ===== 第二步：初始化模型和优化器 =====
    print('\n初始化模型...')
    model  = gineModel.CostModelV2().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01   # L2正则化，防止过拟合
    )

    # 帮助跳出局部最优
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,   # 多少个epoch完成一次余弦周期
        eta_min=1e-6        # 学习率最小值
    )

    # 早停相关变量
    best_val_loss = float('inf')    # 记录历史最好的验证loss
    no_improve    = 0               # 连续多少次验证无改善
    batch_count   = 0               # 全局batch计数器

    # ===== 第三步：训练循环 =====
    print('\n开始训练...')
    print('=' * 50)

    for epoch in range(config.NUM_EPOCHS):
        print(f'\nEpoch {epoch + 1}/{config.NUM_EPOCHS}')

        # 每个epoch开始时打乱训练集索引顺序
        # 目的：让模型每个epoch看到不同顺序的数据，防止过拟合
        random.shuffle(train_indices)

        epoch_train_loss = 0.0      # 记录这个epoch的训练loss
        num_batches      = 0        # 记录这个epoch的batch数量
        stop_training    = False    # 早停标志

        # 按batch_size切分训练集
        for start in range(0, len(train_indices), config.BATCH_SIZE):
            # 取出这个batch的索引
            batch_indices = train_indices[start:start + config.BATCH_SIZE]

            # 训练这个batch，得到loss
            batch_loss = trainAndEval.train_one_batch(
                model, optimizer, file_list, batch_indices
            )
            epoch_train_loss += batch_loss
            num_batches      += 1
            batch_count      += 1

            # 每隔VALIDATE_EVERY个batch验证一次
            if batch_count % config.VALIDATE_EVERY == 0:
                # 随机抽VAL_BATCH_SIZE个验证样本评估
                val_loss =trainAndEval.evaluate(
                    model, file_list, val_indices,
                    max_samples=config.VAL_BATCH_SIZE
                )

                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                print(
                    f'  batch {batch_count:6d} | '
                    f'训练loss: {batch_loss:.4f} | '
                    f'验证loss: {val_loss:.4f} | '
                    f'学习率: {current_lr:.6f}'
                )

                # 学习率调度器根据验证loss调整学习率
                scheduler.step(val_loss)

                # 判断验证loss是否改善
                if val_loss < best_val_loss:
                    # 验证loss改善，保存当前模型
                    best_val_loss = val_loss
                    no_improve    = 0
                    torch.save(model.state_dict(), config.MODEL_SAVE)
                    print(f'  → 验证loss改善至 {best_val_loss:.4f}，模型已保存')
                else:
                    # 验证loss没有改善，计数器加1
                    no_improve += 1
                    print(f'  → 验证loss无改善，已连续 {no_improve}/{config.PATIENCE} 次')

                    # 达到早停阈值，停止训练
                    if no_improve >= config.PATIENCE:
                        print(f'\n早停：连续 {config.PATIENCE} 次验证无改善，停止训练')
                        stop_training = True
                        break

        # 如果触发了早停，退出epoch循环
        if stop_training:
            break

        # 打印这个epoch的平均训练loss
        avg_epoch_loss = epoch_train_loss / num_batches
        print(f'Epoch {epoch + 1} 平均训练loss: {avg_epoch_loss:.4f}')

    # ===== 第四步：最终测试评估 =====
    print('\n' + '=' * 50)
    print('训练结束，开始最终测试...')

    # 加载训练过程中验证loss最好的模型
    model.load_state_dict(torch.load(config.MODEL_SAVE,map_location=device))
    print(f'已加载最优模型（验证loss: {best_val_loss:.4f}）')

    # 在测试集上评估，使用全部测试样本
    test_loss = trainAndEval.evaluate(
        model, file_list, test_indices,
         max_samples=None  # None表示用全部测试样本
    )

    print(f'最终测试loss: {test_loss:.4f}')
    print('=' * 50)

    return model

if __name__ == '__main__':
    train(config.SAMPLE_SAVE)