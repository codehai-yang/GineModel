import random
import torch
import torch.nn as nn
import LoadSample as loadSample


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
        print(f'edge_index最大值: {edge_index.max()}')   # 不能超过174
        print(f'edge_index最小值: {edge_index.min()}')   # 不能小于0
        print(f'节点数量:         {x.shape[0]}')          # 应该是175
        # 转换为tensor
        edge_attr_t, edge_index_t, x_t, y_t = loadSample.sample_to_tensor(
            edge_index,edge_attr,  x, y
        )

        # 前向传播：输入数据，得到预测成本
        pred = model(x_t, edge_index_t, edge_attr_t)

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
                edge_attr, edge_index, x, y
            )

            # 前向传播，得到预测值
            pred = model(x_t, edge_index_t, edge_attr_t)

            # 计算loss并累加
            loss = nn.MSELoss()(pred, y_t)
            total_loss += loss.item()

    # 返回平均loss
    avg_loss = total_loss / len(sampled_indices)
    return avg_loss