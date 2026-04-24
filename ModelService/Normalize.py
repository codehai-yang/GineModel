import numpy as np
import json
import os

_GLOBAL_STATS = {}

def load_global_stats(json_path):
    """
    从 JSON 文件加载全局归一化参数
    """
    global _GLOBAL_STATS
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"归一化参数文件不存在: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        _GLOBAL_STATS = json.load(f)

def normalize_branch_feature(branch_feature):
    """
    对边特征做归一化（使用全局统计量）。
    """
    result = branch_feature.copy()
    length_col = result[:, 3]

    # 使用 .get 提供默认值，防止 KeyError
    mean = _GLOBAL_STATS.get('branch_length_mean', 0)
    std = _GLOBAL_STATS.get('branch_length_std', 1)

    if std > 1e-6:
        result[:, 3] = (length_col - mean) / std
    else:
        result[:, 3] = 0.0

    return result


def normalize_wet_cost(circuit_cost):
    """
    对节点特征矩阵的第176列（湿区成本）做归一化（使用全局统计量）。
    只对非零值做标准化，零（无湿区）保持零。
    """
    result = circuit_cost.copy()
    wet_col = result[:, 175].copy()
    nonzero_mask = wet_col != 0

    if nonzero_mask.sum() > 0:
        mean = _GLOBAL_STATS.get('wet_cost_mean', 0)
        std = _GLOBAL_STATS.get('wet_cost_std', 1)

        if std > 1e-6:
            wet_col[nonzero_mask] = (wet_col[nonzero_mask] - mean) / std
        else:
            wet_col[nonzero_mask] = 1.0

    result[:, 175] = wet_col
    return result


def normalize_price_matrix(circuit_cost):
    """
    对节点特征矩阵的前175列（回路单价矩阵）做归一化（使用全局统计量）。
    """
    result = circuit_cost.copy()
    price_matrix = result[:, :175].copy()
    nonzero_mask = price_matrix != 0

    if nonzero_mask.sum() > 0:
        mean = _GLOBAL_STATS.get('price_mean', 0)
        std = _GLOBAL_STATS.get('price_std', 1)

        if std > 1e-6:
            price_matrix[nonzero_mask] = (price_matrix[nonzero_mask] - mean) / std
        else:
            price_matrix[nonzero_mask] = 1.0

    result[:, :175] = price_matrix
    return result


def normalize_all(branch_feature, circuit_cost):
    """
    对所有需要归一化的字段统一处理。
    """
    # 如果还没加载参数，尝试自动加载默认路径
    # 注意：在多进程 worker 中，_GLOBAL_STATS 初始为空，会触发此加载逻辑
    if not _GLOBAL_STATS:
        # 获取 Pt 目录下的参数文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(current_dir, '..', 'Pt', 'normalization_params.json')
        if os.path.exists(default_path):
            load_global_stats(default_path)
        else:
            raise RuntimeError(f"未找到全局归一化参数文件: {default_path}")

    branch_feature_norm = normalize_branch_feature(branch_feature)
    circuit_cost_norm = normalize_price_matrix(circuit_cost)
    circuit_cost_norm = normalize_wet_cost(circuit_cost_norm)

    return branch_feature_norm, circuit_cost_norm

def verify_normalization(branch_feature_norm, circuit_cost_norm):
    """
    验证归一化结果是否正确。
    """
    print('=== 归一化结果验证 ===')
    length_col = branch_feature_norm[:, 3]
    print(f'\n分支长度（归一化后）:')
    print(f'  均值:   {length_col.mean():.4f}')
    print(f'  标准差: {length_col.std():.4f}')

    onehot = branch_feature_norm[:, :3]
    unique_vals = np.unique(onehot)
    print(f'\n通断状态one-hot: {unique_vals}')

    price_matrix = circuit_cost_norm[:, :175]
    nonzero_price = price_matrix[price_matrix != 0]
    if len(nonzero_price) > 0:
        print(f'\n回路单价（非零值）均值: {nonzero_price.mean():.4f}')

    wet_col = circuit_cost_norm[:, 175]
    nonzero_wet = wet_col[wet_col != 0]
    if len(nonzero_wet) > 0:
        print(f'湿区成本（非零值）均值: {nonzero_wet.mean():.4f}')