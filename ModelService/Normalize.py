import numpy as np


def normalize_branch_feature(branch_feature):
    """
    对边特征做归一化。
    branch_feature [211, 4]：
        前3列：通断状态 one-hot（0或1），不归一化
        第4列：分支长度（连续值），标准化

    参数：
        branch_feature: numpy数组 [211, 4]

    返回：
        归一化后的 branch_feature [211, 4]
    """
    result = branch_feature.copy()          # 复制一份，不修改原始数据

    # 取出第4列：分支长度
    length_col = result[:, 3]               # [211]

    # 分支长度不存在有特殊含义的0（现实中不存在长度=0的分支）
    # 直接对整列做标准化
    mean = length_col.mean()                # 计算这一列的均值
    std  = length_col.std()                 # 计算这一列的标准差

    if std > 0:                             # 避免除以0
        result[:, 3] = (length_col - mean) / std   # 标准化
    else:
        result[:, 3] = 0.0                  # 所有值一样，标准化后全为0

    # 前3列one-hot不做任何处理，直接保留
    return result


def normalize_wet_cost(circuit_cost):
    """
    对节点特征矩阵的第176列（湿区成本）做归一化。
    只对非零值做标准化，零（无湿区）保持零。

    参数：
        circuit_cost: numpy数组 [175, 176]

    返回：
        归一化后的 circuit_cost [175, 176]
    """
    result = circuit_cost.copy()            # 复制一份，不修改原始数据

    # 取出第176列（索引175）：湿区成本
    wet_col = result[:, 175].copy()         # [175]

    # 找出非零位置（有湿区的节点）
    nonzero_mask = wet_col != 0             # bool数组，True表示有湿区

    # 只有存在非零值时才做标准化
    if nonzero_mask.sum() > 0:
        nonzero_vals = wet_col[nonzero_mask]    # 取出所有非零值

        mean = nonzero_vals.mean()              # 非零值的均值
        std  = nonzero_vals.std()               # 非零值的标准差

        if std > 0:                             # 避免除以0
            # 只对非零位置标准化
            wet_col[nonzero_mask] = (nonzero_vals - mean) / std
        else:
            # 所有非零值都一样（比如全是18.8）
            # 标准化后全为0，但和无湿区的0混淆了
            # 改为全部设为1.0，表示有湿区但成本固定
            wet_col[nonzero_mask] = 1.0

    # 零保持零，写回第176列
    result[:, 175] = wet_col

    return result


def normalize_price_matrix(circuit_cost):
    """
    对节点特征矩阵的前175列（回路单价矩阵）做归一化。
    只对非零值做标准化，零（无回路）保持零。

    参数：
        circuit_cost: numpy数组 [175, 176]

    返回：
        归一化后的 circuit_cost [175, 176]
    """
    result = circuit_cost.copy()            # 复制一份，不修改原始数据

    # 取出前175列：回路单价矩阵
    price_matrix = result[:, :175].copy()   # [175, 175]

    # 找出非零位置（有回路的节点对）
    nonzero_mask = price_matrix != 0        # bool数组

    # 只有存在非零值时才做标准化
    if nonzero_mask.sum() > 0:
        nonzero_vals = price_matrix[nonzero_mask]   # 取出所有非零单价值

        mean = nonzero_vals.mean()                  # 所有非零单价的均值
        std  = nonzero_vals.std()                   # 所有非零单价的标准差

        if std > 0:                                 # 避免除以0
            # 只对非零位置标准化
            price_matrix[nonzero_mask] = (nonzero_vals - mean) / std
        else:
            # 所有非零值都一样，标准化后设为1.0
            price_matrix[nonzero_mask] = 1.0

    # 零保持零，写回前175列
    result[:, :175] = price_matrix

    return result


def normalize_all(branch_feature, circuit_cost):
    """
    对所有需要归一化的字段统一处理。
    按顺序执行：
        1. 分支长度标准化（branch_feature第4列）
        2. 回路单价矩阵标准化（circuit_cost前175列）
        3. 湿区成本标准化（circuit_cost第176列）

    参数：
        branch_feature: numpy数组 [211, 4]
        circuit_cost:   numpy数组 [175, 176]

    返回：
        branch_feature_norm: 归一化后的边特征   [211, 4]
        circuit_cost_norm:   归一化后的节点特征  [175, 176]
    """
    # 第一步：归一化分支长度
    branch_feature_norm = normalize_branch_feature(branch_feature)

    # 第二步：归一化回路单价矩阵（前175列）
    circuit_cost_norm = normalize_price_matrix(circuit_cost)

    # 第三步：归一化湿区成本（第176列）
    circuit_cost_norm = normalize_wet_cost(circuit_cost_norm)

    return branch_feature_norm, circuit_cost_norm


def verify_normalization(branch_feature_norm, circuit_cost_norm):
    """
    验证归一化结果是否正确。
    打印各字段归一化后的统计信息。

    参数：
        branch_feature_norm: 归一化后的边特征   [211, 4]
        circuit_cost_norm:   归一化后的节点特征  [175, 176]
    """
    print('=== 归一化结果验证 ===')

    # 验证分支长度（第4列）
    length_col = branch_feature_norm[:, 3]
    print(f'\n分支长度（归一化后）:')
    print(f'  均值:   {length_col.mean():.4f}  （应接近0）')
    print(f'  标准差: {length_col.std():.4f}   （应接近1）')
    print(f'  最大值: {length_col.max():.4f}')
    print(f'  最小值: {length_col.min():.4f}')

    # 验证前3列one-hot（不应该被修改）
    onehot = branch_feature_norm[:, :3]
    unique_vals = np.unique(onehot)
    print(f'\n通断状态one-hot（不应被修改）:')
    print(f'  唯一值: {unique_vals}  （应只有0和1）')

    # 验证回路单价矩阵（前175列）
    price_matrix = circuit_cost_norm[:, :175]
    nonzero_price = price_matrix[price_matrix != 0]
    print(f'\n回路单价矩阵（归一化后非零值）:')
    if len(nonzero_price) > 0:
        print(f'  非零值数量: {len(nonzero_price)}')
        print(f'  均值:       {nonzero_price.mean():.4f}  （应接近0）')
        print(f'  标准差:     {nonzero_price.std():.4f}   （应接近1）')
        print(f'  最大值:     {nonzero_price.max():.4f}')
        print(f'  最小值:     {nonzero_price.min():.4f}')
    else:
        print('  没有非零值')

    # 验证湿区成本（第176列）
    wet_col = circuit_cost_norm[:, 175]
    nonzero_wet = wet_col[wet_col != 0]
    print(f'\n湿区成本（归一化后非零值）:')
    if len(nonzero_wet) > 0:
        print(f'  非零值数量: {len(nonzero_wet)}')
        print(f'  均值:       {nonzero_wet.mean():.4f}  （应接近0）')
        print(f'  标准差:     {nonzero_wet.std():.4f}   （应接近1）')
        print(f'  最大值:     {nonzero_wet.max():.4f}')
        print(f'  最小值:     {nonzero_wet.min():.4f}')
    else:
        print('  没有非零值（所有节点都无湿区）')

    # 验证零值是否保持
    zero_count_price = (circuit_cost_norm[:, :175] == 0).sum()
    zero_count_wet   = (circuit_cost_norm[:, 175]  == 0).sum()
    print(f'\n零值保持验证:')
    print(f'  单价矩阵零值数量: {zero_count_price}  （无回路的位置应保持0）')
    print(f'  湿区成本零值数量: {zero_count_wet}    （无湿区的节点应保持0）')