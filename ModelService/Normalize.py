import numpy as np
import json
import os

_GLOBAL_STATS = {}
_NORMALIZATION_PARAMS_PATH = None

def set_normalization_params_path(path):
    """
    设置归一化参数文件路径（可选，用于命令行指定）
    """
    global _NORMALIZATION_PARAMS_PATH
    _NORMALIZATION_PARAMS_PATH = path

def load_global_stats(json_path=None):
    """
    从 JSON 文件加载全局归一化参数

    参数：
        json_path: 归一化参数文件路径，如果为None则使用默认路径或已设置的路径
    """
    global _GLOBAL_STATS

    if json_path is None:
        json_path = _NORMALIZATION_PARAMS_PATH

    if json_path is None:
        # 使用默认路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, '..', 'Pt', 'normalization_params.json')

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


def normalize_wet_cost(circuit_cost, num_nodes):
    """
    对节点特征矩阵的第 num_nodes 列（湿区成本）做归一化（使用全局统计量）。
    只对非零值做标准化，零（无湿区）保持零。

    参数：
        circuit_cost: numpy [N, F]
        num_nodes:    int, 节点数 N，湿区在第 N 列
    """
    result = circuit_cost.copy()

    # 安全边界检查
    if num_nodes >= result.shape[1]:
        return result  # 如果没有湿区列，直接返回

    wet_col = result[:, num_nodes].copy()
    nonzero_mask = wet_col != 0

    if nonzero_mask.sum() > 0:
        mean = _GLOBAL_STATS.get('wet_cost_mean', 0)
        std = _GLOBAL_STATS.get('wet_cost_std', 1)

        if std > 1e-6:
            wet_col[nonzero_mask] = (wet_col[nonzero_mask] - mean) / std
        else:
            wet_col[nonzero_mask] = 1.0

    result[:, num_nodes] = wet_col
    return result


def normalize_price_matrix(circuit_cost, num_nodes):
    """
    对节点特征矩阵的前 num_nodes 列（回路单价矩阵）做归一化（使用全局统计量）。

    参数：
        circuit_cost: numpy [N, F]
        num_nodes:    int, 节点数 N，价格矩阵在第 0 ~ N-1 列
    """
    result = circuit_cost.copy()
    price_matrix = result[:, :num_nodes].copy()
    nonzero_mask = price_matrix != 0

    if nonzero_mask.sum() > 0:
        mean = _GLOBAL_STATS.get('price_mean', 0)
        std = _GLOBAL_STATS.get('price_std', 1)

        if std > 1e-6:
            price_matrix[nonzero_mask] = (price_matrix[nonzero_mask] - mean) / std
        else:
            price_matrix[nonzero_mask] = 1.0

    result[:, :num_nodes] = price_matrix
    return result


def normalize_y(y):
    """
    对样本标签 y（总成本）做标准化。
    使用 JSON 中的 total_cost_mean / total_cost_std。

    参数：
        y : float（总成本原始值）
    返回：
        float（标准化后的 y）
    """
    # 如果还没加载参数，尝试自动加载
    if not _GLOBAL_STATS:
        load_global_stats()

    mean = _GLOBAL_STATS.get('total_cost_mean', 0.0)
    std  = _GLOBAL_STATS.get('total_cost_std', 1.0)

    if std > 1e-6:
        return (y - mean) / std
    return 0.0


def denormalize_y(y_norm):
    """
    将标准化后的 y 还原为原始成本值。

    参数：
        y_norm : float 或 numpy array / tensor（标准化后的值）
    返回：
        同类型，还原后的原始成本
    """
    if not _GLOBAL_STATS:
        load_global_stats()

    mean = _GLOBAL_STATS.get('total_cost_mean', 0.0)
    std  = _GLOBAL_STATS.get('total_cost_std', 1.0)

    return y_norm * std + mean


def normalize_all(branch_feature, circuit_cost, y, num_nodes):
    """
    对所有需要归一化的字段统一处理。
    包含：分支长度、回路单价（前 num_nodes 列）、湿区成本（第 num_nodes 列）、样本标签 y。

    参数：
        branch_feature: numpy [E, 4] 边特征
        circuit_cost:   numpy [N, F] 节点特征矩阵
        y:              float         标签
        num_nodes:      int           节点数 N（决定回路单价和湿区的列位置）

    返回：
        (branch_feature_norm, circuit_cost_norm, y_norm)
    """
    # 如果还没加载参数，尝试自动加载
    if not _GLOBAL_STATS:
        load_global_stats()

    branch_feature_norm = normalize_branch_feature(branch_feature)
    circuit_cost_norm = normalize_price_matrix(circuit_cost, num_nodes)
    circuit_cost_norm = normalize_wet_cost(circuit_cost_norm, num_nodes)

    # 标签 y 标准化
    y_norm = normalize_y(y)

    return branch_feature_norm, circuit_cost_norm, y_norm


def verify_normalization(branch_feature_norm, circuit_cost_norm, num_nodes):
    """
    验证归一化结果是否正确。

    参数：
        branch_feature_norm: numpy [E, 4]
        circuit_cost_norm:   numpy [N, F]
        num_nodes:           int, 节点数
    """
    print('=== 归一化结果验证 ===')
    length_col = branch_feature_norm[:, 3]
    print(f'\n分支长度（归一化后）:')
    print(f'  均值:   {length_col.mean():.4f}')
    print(f'  标准差: {length_col.std():.4f}')

    onehot = branch_feature_norm[:, :3]
    unique_vals = np.unique(onehot)
    print(f'\n通断状态one-hot: {unique_vals}')

    # 回路单价：前 num_nodes 列
    price_matrix = circuit_cost_norm[:, :num_nodes]
    nonzero_price = price_matrix[price_matrix != 0]
    if len(nonzero_price) > 0:
        print(f'\n回路单价（非零值）均值: {nonzero_price.mean():.4f}')

    # 湿区成本：第 num_nodes 列
    if num_nodes < circuit_cost_norm.shape[1]:
        wet_col = circuit_cost_norm[:, num_nodes]
        nonzero_wet = wet_col[wet_col != 0]
        if len(nonzero_wet) > 0:
            print(f'湿区成本（非零值）均值: {nonzero_wet.mean():.4f}')

    # 填充列：第 num_nodes+1 列到最后一列
    if num_nodes + 1 < circuit_cost_norm.shape[1]:
        padding_cols = circuit_cost_norm[:, num_nodes + 1:]
        if padding_cols.size > 0:
            print(f'填充列（第 {num_nodes + 1} 列之后）全为 0: {np.all(padding_cols == 0)}')


def normalize_node_features(price_matrix, wet_costs,
                            price_min, price_max,
                            wet_min,   wet_max):
    """
    对节点特征矩阵做归一化。
    0 值保持 0，只对非 0 值归一化。

    参数：
        price_matrix: numpy [N, N] 回路单价矩阵
        wet_costs:    numpy [N]    每个节点的湿区成本
        price_min:    float
        price_max:    float
        wet_min:      float
        wet_max:      float

    返回：
        归一化后的 (price_matrix, wet_costs)
    """
    # 归一化回路单价（排除 0）
    price_range = price_max - price_min
    if price_range > 1e-6:
        nonzero_mask = price_matrix != 0
        price_matrix = np.where(
            nonzero_mask,
            (price_matrix - price_min) / price_range,
            0.0
        )

    # 归一化湿区成本（排除 0）
    wet_range = wet_max - wet_min
    if wet_range > 1e-6:
        nonzero_mask = wet_costs != 0
        wet_costs = np.where(
            nonzero_mask,
            (wet_costs - wet_min) / wet_range,
            0.0
        )

    return price_matrix, wet_costs