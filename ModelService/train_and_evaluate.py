import random
import torch
import torch.nn as nn
import LoadSample as loadSample
import pandas as pd


def train_one_batch(model, optimizer, file_list, batch_indices):
    """
    训练一个batch：前向传播 + 计算loss + 反向传播 + 更新权重。

    参数：
        model:           GINEClassifier模型
        optimizer:       优化器
        file_list:       数据文件路径列表
        batch_indices:   这个batch的样本索引列表，每个元素是(file_idx, sample_idx)
    返回：
        batch_loss: 这个batch的平均loss值（float）
    """
    model.train()               # 切换到训练模式，启用BatchNorm和Dropout
    optimizer.zero_grad()       # 清空上一个batch的梯度，防止梯度累积

    batch_loss = torch.tensor(0.0)  # 累积这个batch所有样本的loss

    for file_idx, sample_idx in batch_indices:
        # 从文件中读取这个样本
        edge_index,edge_attr, x, y = loadSample.read_sample_by_index(
            file_list, file_idx, sample_idx
        )
        # 转换为tensor
        edge_index_t,edge_attr_t,  x_t, y_t = loadSample.sample_to_tensor(
            edge_index,edge_attr,  x, y
        )

        # 前向传播：输入数据，得到预测成本
        pred = model(x_t, edge_index_t, edge_attr_t)
        print("预测成本:" +  str(pred.item()))
        print("真实成本:" +  str(y_t.item()))

        # 计算MSE损失：预测值和真实值的均方误差
        loss = nn.MSELoss()(pred, y_t)

        # 累加这个样本的loss
        batch_loss = batch_loss + loss

    # 对batch内所有样本的loss求平均
    batch_loss = batch_loss / len(batch_indices)

    # 反向传播：计算所有参数的梯度
    batch_loss.backward()

    # 梯度裁剪：防止梯度爆炸，限制梯度的最大范数为1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新模型参数：用梯度更新权重
    optimizer.step()

    return batch_loss.item()    # 返回python float，不再需要梯度


def evaluate(model, file_list, eval_indices, max_samples=None):
    """
    在给定数据集上评估模型，计算平均loss。
    评估时不更新模型权重。

    参数：
        model:           CostModel模型
        file_list:       数据文件路径列表
        eval_indices:    评估集的样本索引列表
        circuit_records: 回路记录列表
        max_samples:     最多评估多少个样本，None表示全部评估

    返回：
        avg_loss: 平均MSE损失（float）
    """
    model.eval()    # 切换到评估模式，关闭BatchNorm的训练行为

    # 如果指定了最大样本数，随机抽取
    if max_samples and max_samples < len(eval_indices):
        sampled_indices = random.sample(eval_indices, max_samples)
    else:
        sampled_indices = eval_indices  # 使用全部评估样本

    total_loss = 0.0

    # torch.no_grad()：评估时不计算梯度，节省内存和计算
    with torch.no_grad():
        for file_idx, sample_idx in sampled_indices:
            # 读取样本
            edge_index,edge_attr,  x, y = loadSample.read_sample_by_index(
                file_list, file_idx, sample_idx
            )

            # 转换为tensor
            edge_index_t,edge_attr_t,  x_t, y_t = loadSample.sample_to_tensor(
                edge_index,edge_attr,  x, y
            )

            # 前向传播，得到预测值
            pred = model(x_t, edge_index_t, edge_attr_t)

            # 计算loss并累加
            loss = nn.MSELoss()(pred, y_t)
            total_loss += loss.item()

    # 返回平均loss
    avg_loss = total_loss / len(sampled_indices)
    return avg_loss


def evaluate_and_save_results(model, file_list, eval_indices, save_path, max_samples=None):
    """
    评估模型并将预测结果和真实值保存到 Excel。

    参数：
        model:          CostModel 模型
        file_list:      数据文件路径列表
        eval_indices:   评估集的样本索引列表
        save_path:      Excel 保存路径
        max_samples:    最多评估多少个样本，None 表示全部评估
    """
    model.eval()

    # 如果指定了最大样本数，随机抽取
    if max_samples and max_samples < len(eval_indices):
        sampled_indices = random.sample(eval_indices, max_samples)
    else:
        sampled_indices = eval_indices

    results = []  # 存储结果的列表

    print(f'\n开始在评估集上预测并保存结果...')
    print(f'样本数量：{len(sampled_indices)}')

    with torch.no_grad():
        for idx, (file_idx, sample_idx) in enumerate(sampled_indices):
            # 读取样本
            edge_index, edge_attr, x, y = loadSample.read_sample_by_index(
                file_list, file_idx, sample_idx
            )

            # 转换为 tensor
            edge_index_t, edge_attr_t, x_t, y_t = loadSample.sample_to_tensor(
                edge_index, edge_attr, x, y
            )

            # 前向传播，得到预测值
            pred = model(x_t, edge_index_t, edge_attr_t)

            # 收集结果（保留原始精度）
            results.append({
                '样本编号': idx + 1,
                '样本索引': sample_idx,
                '预测成本': pred.item(),
                '真实成本': y_t.item(),
                '误差': abs(pred.item() - y_t.item())
            })

            # 每 50 个样本打印一次进度
            if (idx + 1) % 50 == 0:
                print(f'  已处理 {idx + 1}/{len(sampled_indices)} 个样本')

    # 创建 DataFrame 并保存到 Excel
    df = pd.DataFrame(results)

    # 调整列顺序
    df = df[['样本编号', '样本索引', '预测成本', '真实成本', '误差']]

    # 保存到 Excel
    df.to_excel(save_path, index=False, float_format='%.6f')

    print(f'\n结果已保存到：{save_path}')
    print(f'总样本数：{len(df)}')
    print(f'平均误差：{df["误差"].mean():.4f}')
    return df