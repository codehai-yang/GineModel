import struct
import numpy as np
import torch
import GINEClassifier as gineModel
import GlobalConfig as config
import Normalize as nz
import os
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = gineModel.CostModelV2()
model_path = os.path.join(config.MODEL_SAVE)
print(f'正在加载模型: {model_path}')

# 加载前先检查维度是否匹配
try:
    state = torch.load(model_path, map_location='cpu')
    if 'input_proj.weight' in state:
        saved_dim = state['input_proj.weight'].shape[1]
        expected_dim = config.NODE_FEAT_DIM
        if saved_dim != expected_dim:
            raise RuntimeError(
                f'\n' +
                f'=' * 60 + '\n' +
                f'❌ 模型维度不匹配!\n' +
                f'   保存的模型节点特征维度: {saved_dim}\n' +
                f'   config.NODE_FEAT_DIM:   {expected_dim}\n' +
                f'   解决方案: 重新训练模型 (del Pt\\best_model.pt && python train_gine.py)\n' +
                f'=' * 60
            )
    del state  # 释放内存
except RuntimeError:
    raise
except Exception as e:
    print(f'⚠️ 模型检查时出错: {e}')

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f'✅ 模型加载成功')

def predict_single(model, filepath):
    """
    读取单个二进制样本文件，适配新格式（头部8字节 + 动态N/E）。
    在Python端做标准化（与训练时一致），然后推理。

    参数：
        model:    GINE 模型
        filepath: 单个样本文件路径

    返回：
        float: 预测成本（原始空间）
    """
    with open(filepath, 'rb') as f:
        # 1. 读取头部：N 和 E
        header = f.read(config.HEADER_BYTES)
        if len(header) < config.HEADER_BYTES:
            raise ValueError(f"文件 {filepath} 头部不足 {config.HEADER_BYTES} 字节")

        N, E = struct.unpack('>ii', header)

        # 2. 按动态 N/E 读取 edge_index [2, E]
        edge_index_bytes = E * 2 * 4
        edge_index = np.frombuffer(
            f.read(edge_index_bytes), dtype='>i4'
        ).reshape(2, E).copy().astype('<i4')

        # 3. 读取 edge_attr [E, 4]
        edge_attr_bytes = E * config.EDGE_FEAT_DIM * 4
        edge_attr = np.frombuffer(
            f.read(edge_attr_bytes), dtype='>f4'
        ).reshape(E, config.EDGE_FEAT_DIM).copy().astype('<f4')

        # 4. 读取 x [N, 200]
        x_bytes = N * config.NODE_FEAT_DIM * 4
        x = np.frombuffer(
            f.read(x_bytes), dtype='>f4'
        ).reshape(N, config.NODE_FEAT_DIM).copy().astype('<f4')

        print(f"x 形状: {x.shape}")
        print(f"x 第 0 行非零列索引: {np.where(x[0] != 0)[0]}")
        print(f"x 第 6 行 185-199 列: {x[6, 189:200]}")
        # 5. 读取 y: 总成本、总重量、总长度（从后 12 字节读取原始值）
        # y_bytes = f.read(config.Y_FEAT_COUNT * 4)
        # total_cost, total_weight, total_length = struct.unpack('>fff', y_bytes)

    # 5. 标准化（与训练时一致）
    # 传入 N：回路单价前 N 列，湿区第 N 列
    edge_attr, x, _ = nz.normalize_all(edge_attr, x, 0.0, N)  # y 占位，推理时不需要

    print("标准化后的 x 矩阵 (前 10 行):")
    print(x[:10, :])
    # 6. 转 tensor
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(device)
    edge_attr_t  = torch.tensor(edge_attr,  dtype=torch.float).to(device)
    x_t          = torch.tensor(x,          dtype=torch.float).to(device)

    with torch.no_grad():
        pred_norm = model(x_t, edge_index_t, edge_attr_t)  # 标准化空间

    # 7. 反标准化回原始成本
    pred_real = nz.denormalize_y(pred_norm.cpu().item())

    return pred_real


# 使用
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GINE Model Inference')
    parser.add_argument('--data_dir', type=str, default=r'F:\office\pythonProjects\GINEModel\javaTest', help='样本文件所在目录')
    parser.add_argument('--sample', type=str, default='predict_input.bin', help='样本文件名')
    args = parser.parse_args()

    # 优先使用命令行传入的 data_dir，否则使用 config.SAMPLE_SAVE
    data_dir = args.data_dir if args.data_dir else config.SAMPLE_SAVE
    test_path = os.path.join(data_dir, args.sample)
    pred_cost = predict_single(model, test_path)


    print('=' * 60)
    print(f'预测成本: {pred_cost:.4f}')
    print('=' * 60)

    import torch

    # 替换为你的模型路径
    model_path = r"F:\office\pythonProjects\GineService\best_model.pt"

    # 加载权重（不加载模型结构，只看权重）
    state_dict = torch.load(model_path, map_location='cpu')

    # 查看输入投影层的权重形状
    input_proj_weight = state_dict['input_proj.weight']
    print(f"input_proj.weight 形状: {input_proj_weight.shape}")
    print(f"模型支持的输入维度 (node_feat_dim): {input_proj_weight.shape[1]}")
