import math
import torch
import torch.nn as nn
import pandas as pd
import Normalize as nz


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
def evaluate_and_save_results(model, loader, save_path, device, max_samples=None, hyperparams=None):
    """
    评估模型并将预测结果保存到 Excel（四个 Sheet）。

    Sheet1-预测明细: 样本编号、预测成本、真实成本、误差(带正负)、误差百分比
    Sheet2-整体指标: 误差均值、误差绝对值均值、误差百分比均值、误差百分比绝对值均值
    Sheet3-误差分布: 按1%区间动态展开，正负分开统计，覆盖所有样本
    Sheet4-训练超参数: 模型和训练相关的超参数一览

    参数：
        model       : CostModelV2
        loader      : DataLoader（测试集）
        save_path   : Excel 保存路径
        device      : 目标设备
        max_samples : None 表示全部
        hyperparams : dict, 训练超参数（可选）
    """
    model.eval()

    results     = []
    sample_idx  = 0

    print(f'\n开始预测并保存结果...')

    for batch in loader:
        batch = batch.to(device)

        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)  # [B] 标准化空间
        ys    = batch.y.squeeze()                                                # [B] 标准化空间

        preds = preds.view(-1)
        ys    = ys.view(-1)

        # 反归一化：还原到原始成本（给人看的）
        preds_real = nz.denormalize_y(preds.detach().cpu())
        ys_real    = nz.denormalize_y(ys.detach().cpu())

        for pred_val, y_val in zip(preds_real.tolist(), ys_real.tolist()):
            err = pred_val - y_val
            pct = (err / y_val * 100) if y_val != 0 else 0.0
            results.append({
                '样本编号': sample_idx + 1,
                '预测成本': pred_val,
                '真实成本': y_val,
                '误差'    : err,
                '误差百分比': pct
            })
            sample_idx += 1

        if (sample_idx) % 100 == 0:
            print(f'  已处理 {sample_idx} 个样本')

        if max_samples is not None and sample_idx >= max_samples:
            break

    df = pd.DataFrame(results)[['样本编号', '预测成本', '真实成本', '误差', '误差百分比']]
    pct_series = df['误差百分比']
    err_series = df['误差']

    # ---------- Sheet 2: 整体指标 ----------
    summary = {
        '指标': ['误差均值', '误差绝对值均值', '误差百分比均值(%)', '误差百分比绝对值均值(%)'],
        '数值': [
            err_series.mean(),
            err_series.abs().mean(),
            pct_series.mean(),
            pct_series.abs().mean()
        ]
    }
    df_summary = pd.DataFrame(summary)

    # ---------- Sheet 3: 误差分布（动态区间，覆盖所有样本）----------
    min_pct = math.floor(pct_series.min()) if len(pct_series) > 0 else -5
    max_pct = math.ceil(pct_series.max()) if len(pct_series) > 0 else 5

    dist_data = []
    for low in range(min_pct, max_pct):
        high = low + 1
        if low < 0:
            label = f'{low}%~{high}%'  # 负区间左开右闭更直观
        else:
            label = f'{low}%~{high}%'
        cnt = ((pct_series >= low) & (pct_series < high)).sum()
        dist_data.append({'误差区间': label, '样本数': cnt})

    df_dist = pd.DataFrame(dist_data)

    # ---------- Sheet 4: 训练超参数 ----------
    df_params = pd.DataFrame()
    if hyperparams:
        df_params = pd.DataFrame({
            '超参数': list(hyperparams.keys()),
            '值'   : list(hyperparams.values())
        })

    # ---------- 写入 Excel ----------
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='预测明细', index=False, float_format='%.6f')
        df_summary.to_excel(writer, sheet_name='整体指标', index=False, float_format='%.6f')
        df_dist.to_excel(writer, sheet_name='误差分布', index=False)
        if hyperparams:
            df_params.to_excel(writer, sheet_name='训练超参数', index=False)

    print(f'\n结果已保存到：{save_path}')
    print(f'总样本数：{len(df)}')
    print(f'误差均值：{err_series.mean():.4f}  (正=预测偏高, 负=预测偏低)')
    print(f'误差绝对值均值：{err_series.abs().mean():.4f}')
    print(f'误差百分比均值：{pct_series.mean():.2f}%')
    print(f'误差百分比绝对值均值：{pct_series.abs().mean():.2f}%')
    return df