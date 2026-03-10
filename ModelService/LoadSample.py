import os
import GlobalConfig as config       #全局配置文件
import os
import random
import struct
import numpy as np
import torch
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

def build_global_indices(data_dir):
    """
    扫描指定目录下所有二进制样本文件，建立全局索引。

    参数：
        data_dir: 存放样本文件的目录路径，比如 './data/samples'

    返回：
        all_indices: 全局索引列表，每个元素是 (file_idx, sample_idx)
        file_list:   扫描到的文件路径列表，供后续读取使用
    """
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'目录不存在: {data_dir}')

    # 匹配目录下所有文件，排除子目录
    file_list = sorted([
        f for f in glob.glob(os.path.join(data_dir, '*'))
        if os.path.isfile(f)    # 只保留文件
    ])

    if len(file_list) == 0:
        raise FileNotFoundError(f'目录 {data_dir} 下没有找到任何文件')

    print(f'目录 {data_dir} 下找到 {len(file_list)} 个文件')

    all_indices = []

    for file_idx, filepath in enumerate(file_list):
        # 通过文件大小计算样本数量
        file_size   = os.path.getsize(filepath)
        num_samples = file_size // config.SAMPLE_BYTES

        # 验证文件大小是否正确
        if file_size % config.SAMPLE_BYTES != 0:
            raise ValueError(
                f'文件 {filepath} 大小异常: '
                f'{file_size} 字节不能被 {config.SAMPLE_BYTES} 整除'
            )

        # 把这个文件的所有样本索引加入全局列表
        for sample_idx in range(num_samples):
            all_indices.append((file_idx, sample_idx))

        print(f'  {os.path.basename(filepath)}: {num_samples} 个样本')

    print(f'总样本数: {len(all_indices)}')

    # 同时返回file_list，后续read_sample_by_index需要用它找文件路径
    return all_indices, file_list

def split_indices(all_indices, train_ratio=0.7, val_ratio=0.15, seed=config.RANDOM_SEED):
    """
    将全局索引划分为训练集、验证集、测试集。
    划分方式：只操作索引，不读取任何数据，内存占用极小。

    参数：
        all_indices:  全局索引列表
        train_ratio:  训练集比例，默认70%
        val_ratio:    验证集比例，默认15%
        seed:         随机种子，保证每次划分结果一致

    返回：
        train_indices, val_indices, test_indices
    """
    # 固定随机种子，保证每次运行划分结果相同
    random.seed(seed)

    # 打乱索引顺序，确保随机分配
    indices = all_indices.copy()
    random.shuffle(indices)

    # 计算各集合的大小
    total      = len(indices)
    train_size = int(total * train_ratio)
    val_size   = int(total * val_ratio)
    # 测试集取剩余的所有样本

    # 按比例切分
    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size + val_size]
    test_indices  = indices[train_size + val_size:]

    print(f'训练集: {len(train_indices)} 个样本')
    print(f'验证集: {len(val_indices)}   个样本')
    print(f'测试集: {len(test_indices)}  个样本')

    return train_indices, val_indices, test_indices

def read_sample(f):
    """
    从已打开的文件对象中读取一个样本。
    读取顺序必须和Java写入顺序完全一致。

    参数：
        f: 已打开的二进制文件对象，文件指针需提前seek到正确位置

    返回：
        edge_attr:  numpy数组 [211, 4]  边特征：通断(3维) + 分支长度(1维)
        edge_index: numpy数组 [2, 211]  边索引：起点行 + 终点行
        x:          numpy数组 [175, 1]  节点特征：是否湿区
        y:          float               标签：总成本
    """
    # # 读取 edge_index [2, 211]，int32 类型
    # edge_index_bytes = f.read(config.EDGE_INDEX_BYTES)
    #
    # # 尝试不同的数据类型和字节序
    # edge_index = np.frombuffer(
    #     edge_index_bytes,
    #     dtype='<i4'  # 显式指定小端 int32
    # ).reshape(2, config.NUM_BRANCHES)
    #
    # # 检查数据是否合理
    # if edge_index.max() >= config.NUM_NODES or edge_index.min() < 0:
    #     print(f'警告：edge_index 异常 [{edge_index.min()}, {edge_index.max()}]')
    #     print(f'原始字节 (前 20): {edge_index_bytes[:20].hex()}')
    #
    #     # 尝试按 float32 读取看看
    #     edge_index_as_float = np.frombuffer(
    #         edge_index_bytes,
    #         dtype='<f4'
    #     ).reshape(2, config.NUM_BRANCHES)
    #     print(f'如果按 float32 解析：[{edge_index_as_float.min():.4f}, {edge_index_as_float.max():.4f}]')

    # 读取 edge_index [2, 211]，int32类型
    # 2行(起点/终点) × 211条分支 × 4字节 = 1688字节
    edge_index = np.frombuffer(
        f.read(config.EDGE_INDEX_BYTES),   # 读取1688字节
        dtype='>i4'               # 按int32解析，节点索引是整数
    ).reshape(2,  config.NUM_BRANCHES)      # 重塑为 [2, 211]
    edge_index = edge_index.astype('<i4')

    # 读取 edge_attr [211, 4]，float32类型
    # 211条分支 × 4个特征 × 4字节 = 3376字节
    edge_attr = np.frombuffer(
        f.read(config.EDGE_ATTR_BYTES),    # 读取3376字节
        dtype='>f4'             # 按float32解析
    ).reshape(config.NUM_BRANCHES,config.EDGE_FEAT_DIM)  # 重塑为 [211, 4]
    edge_attr = edge_attr.astype('<f4')



    # 读取 x [175, 1]，float32类型
    # 175个节点 × 1个特征 × 4字节 = 700字节
    x = np.frombuffer(
        f.read(config.X_BYTES),            # 读取700字节
        dtype='>f4'             # 按float32解析
    ).reshape(config.NUM_NODES,config.NODE_FEAT_DIM)  # 重塑为 [175, 176]
    x = x.astype('<f4')

    # 读取 y，单个float32
    # 12字节，总成本，总长度，总重量
    cost, = struct.unpack('>f', f.read(4))       #总成本训练用
    weight, = struct.unpack('>f', f.read(4))     #总重量，暂时不用
    length, = struct.unpack('>f', f.read(4))     #总长度，暂时不用
    return  edge_index,edge_attr, x, round(cost,2)


def read_sample_by_index(file_list, file_idx, sample_idx):
    """
    根据全局索引读取指定样本。
    使用seek直接跳到样本位置，不需要从头读取。

    参数：
        file_list:   数据文件路径列表
        file_idx:    文件编号
        sample_idx:  文件内样本编号

    返回：
        edge_attr, edge_index, x, y（同read_sample）
    """
    filepath = file_list[file_idx]          # 找到对应文件路径

    with open(filepath, 'rb') as f:
        # 计算这个样本在文件中的字节偏移量
        # 第0个样本从字节0开始，第1个从SAMPLE_BYTES开始，以此类推
        offset = sample_idx * config.SAMPLE_BYTES

        f.seek(offset)                      # 直接跳到该样本的起始位置
        return read_sample(f)               # 从当前位置读取一个样本


def sample_to_tensor(edge_index,edge_attr,  x, y):
    """
    将numpy数组转换为PyTorch tensor，供模型使用。

    参数：
        edge_index: numpy [2, 211]
        edge_attr:  numpy [211, 4]
        x:          numpy [175, 176]
        y:          float

    返回：
        对应的torch tensor，数据类型正确
    """
    # edge_index转long tensor，PyG要求边索引必须是long(int64)类型
    edge_index_t = torch.tensor(edge_index,dtype=torch.long).to(device)
    # edge_attr转float tensor，模型计算需要float类型
    edge_attr_t  = torch.tensor(edge_attr,dtype=torch.float).to(device)



    # x转float tensor
    x_t          = torch.tensor(x,dtype=torch.float).to(device)

    # y转float tensor，包在列表里变成[1]维tensor
    y_t          = torch.tensor([y],dtype=torch.float).to(device)

    return edge_index_t,edge_attr_t,  x_t, y_t

def normalize_node_features(price_matrix, wet_costs,
                            price_min, price_max,
                            wet_min,   wet_max):
    """
    对175×176节点特征矩阵做归一化。
    0值保持0，只对非0值归一化。

    参数：
        price_matrix: [175, 175] 回路单价矩阵
        wet_costs:    [175, 1]   湿区成本列
        price_min/max: 回路单价的归一化范围
        wet_min/max:   湿区成本的归一化范围
    """
    # 复制一份，不修改原始数据
    price_norm = price_matrix.copy()
    wet_norm   = wet_costs.copy()

    # 只对非0位置归一化，0保持0
    mask_price = price_matrix != 0          # 找出非0位置
    price_norm[mask_price] = (
                                     price_matrix[mask_price] - price_min
                             ) / (price_max - price_min)

    mask_wet = wet_costs != 0
    wet_norm[mask_wet] = (
                                 wet_costs[mask_wet] - wet_min
                         ) / (wet_max - wet_min)

    # 拼接成[175, 176]
    x = np.concatenate([price_norm, wet_norm], axis=1)

    return x