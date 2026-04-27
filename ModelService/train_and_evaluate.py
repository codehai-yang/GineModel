import torch
import torch.nn as nn
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  核心训练/评估函数
#  调用方只需传入 DataLoader，不再需要 file_list / indices
# ─────────────────────────────────────────────────────────────

def train_one_batch(model, optimizer, batch, device):
    """
    训练一个已经由 DataLoader 打包好的 batch。

    DataLoader 通过 torch_geometric.data.Batch.from_data_list 把多个图
    拼成一张大图传进来，batch.batch 向量记录每个节点属于哪张图，
    global_add_pool 用它把节点嵌入正确地归约到各自的图。

    参数：
        model     : CostModelV2
        optimizer : 优化器
        batch     : torch_geometric.data.Batch，包含 x / edge_index / edge_attr / y / batch
        device    : 目标设备

    返回：
        (loss_sum, num_samples) 元组，用于后续加权平均
    """
    model.train()
    optimizer.zero_grad()

    batch = batch.to(device)

    pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # [B]
    y    = batch.y.squeeze()                                                # [B]

    loss = nn.MSELoss(reduction='sum')(pred, y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), batch.num_graphs


@torch.no_grad()
def evaluate(model, loader, device, max_samples=None):
    """
    在 DataLoader 上评估模型，返回平均 MSE。

    参数：
        model       : CostModelV2
        loader      : torch_geometric.loader.DataLoader（验证集或测试集）
        device      : 目标设备
        max_samples : 最多评估多少个样本，None 表示全部；
                      用于验证阶段快速采样，不影响 loader 本身

    返回：
        avg_loss（float）
    """
    model.eval()

    total_loss  = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y    = batch.y.squeeze()

        # MSE 是对所有样本求均值，这里用 sum 后统一平均
        loss = nn.MSELoss(reduction='sum')(pred, y)
        total_loss  += loss.item()
        total_count += batch.num_graphs

        if max_samples is not None and total_count >= max_samples:
            break

    return total_loss / total_count


@torch.no_grad()
def evaluate_and_save_results(model, loader, save_path, device, max_samples=None):
    """
    评估模型并将预测结果保存到 Excel。

    参数：
        model       : CostModelV2
        loader      : DataLoader（测试集）
        save_path   : Excel 保存路径
        device      : 目标设备
        max_samples : None 表示全部
    """
    model.eval()

    results     = []
    sample_idx  = 0

    print(f'\n开始预测并保存结果...')

    for batch in loader:
        batch = batch.to(device)

        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # [B]
        ys    = batch.y.squeeze()                                                # [B]

        # 兼容 batch_size=1 时 squeeze 成标量的情况
        preds = preds.view(-1)
        ys    = ys.view(-1)

        for pred_val, y_val in zip(preds.tolist(), ys.tolist()):
            results.append({
                '样本编号': sample_idx + 1,
                '预测成本': pred_val,
                '真实成本': y_val,
                '误差'    : abs(pred_val - y_val)
            })
            sample_idx += 1

        if (sample_idx) % 100 == 0:
            print(f'  已处理 {sample_idx} 个样本')

        if max_samples is not None and sample_idx >= max_samples:
            break

    df = pd.DataFrame(results)[['样本编号', '预测成本', '真实成本', '误差']]
    df.to_excel(save_path, index=False, float_format='%.6f')

    print(f'\n结果已保存到：{save_path}')
    print(f'总样本数：{len(df)}')
    print(f'平均误差：{df["误差"].mean():.4f}')
    return df