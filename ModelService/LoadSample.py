import GlobalConfig as config       #全局配置文件
import os
import random
import struct
import numpy as np
import torch
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


def calc_sample_bytes(N, E):
    """
    根据节点数 N 和边数 E 计算样本的字节大小。

    样本结构：
        N(int32) + E(int32)                              = 8 字节（头部）
        edge_index: E * 2 个 int32                        = E * 8 字节
        edge_attr:  E * 4 个 float32                      = E * 16 字节
        x:          N * 200 个 float32（固定200维）         = N * 800 字节
        y:          3 个 float32                          = 12 字节
        ─────────────────────────────────────────────────────────
        总计: 8 + E*24 + N*800 + 12 = 20 + E*24 + N*800
    """
    return config.HEADER_BYTES + \
           (E * 2 * 4) + \
           (E * config.EDGE_FEAT_DIM * 4) + \
           (N * config.NODE_FEAT_DIM * 4) + \
           (config.Y_FEAT_COUNT * 4)


def build_sample_index(data_dir):
    """
    扫描 data_dir 下所有二进制样本文件，逐样本读取头部（N, E），
    建立每个样本在文件中的起始偏移量索引。

    参数：
        data_dir: 样本文件目录

    返回：
        all_indices: [(file_idx, sample_idx, offset, N, E), ...]
        file_list:   文件路径列表
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'目录不存在: {data_dir}')

    file_list = sorted([
        f for f in glob.glob(os.path.join(data_dir, '*'))
        if os.path.isfile(f)
    ])

    if len(file_list) == 0:
        raise FileNotFoundError(f'目录 {data_dir} 下没有找到任何文件')

    print(f'目录 {data_dir} 下找到 {len(file_list)} 个文件')

    all_indices = []

    for file_idx, filepath in enumerate(file_list):
        file_size = os.path.getsize(filepath)
        sample_count = 0

        with open(filepath, 'rb') as f:
            offset = 0

            while offset + config.HEADER_BYTES <= file_size:
                # 读取头部 N、E
                header = f.read(config.HEADER_BYTES)
                if len(header) < config.HEADER_BYTES:
                    break

                N, E = struct.unpack('>ii', header)

                # 计算这个样本的总字节数
                sample_bytes = calc_sample_bytes(N, E)

                # 过滤不合理的样本（防止死循环）
                if N <= 0 or E <= 0 or offset + sample_bytes > file_size:
                    break

                all_indices.append((file_idx, sample_count, offset, N, E))
                sample_count += 1

                # 跳到下一个样本的起始位置
                offset += sample_bytes
                f.seek(offset)

        print(f'  {os.path.basename(filepath)}: {sample_count} 个样本')

    print(f'总样本数: {len(all_indices)}')
    return all_indices, file_list


def split_indices(all_indices, train_ratio=0.7, val_ratio=0.15, seed=config.RANDOM_SEED):
    """
    将全局索引划分为训练集、验证集、测试集。
    只操作索引，不读取任何数据。

    参数：
        all_indices:  [(file_idx, sample_idx, offset, N, E), ...]
        train_ratio:  训练集比例，默认 70%
        val_ratio:    验证集比例，默认 15%
        seed:         随机种子

    返回：
        train_indices, val_indices, test_indices
    """
    random.seed(seed)

    indices = all_indices.copy()
    random.shuffle(indices)

    total      = len(indices)
    train_size = int(total * train_ratio)
    val_size   = int(total * val_ratio)

    train_indices = indices[:train_size]
    val_indices   = indices[train_size:train_size + val_size]
    test_indices  = indices[train_size + val_size:]

    print(f'训练集: {len(train_indices)} 个样本')
    print(f'验证集: {len(val_indices)}   个样本')
    print(f'测试集: {len(test_indices)}  个样本')

    return train_indices, val_indices, test_indices


def read_sample_from_file(filepath, offset, N, E):
    """
    从文件中按偏移量读取一个样本，节点特征 x 固定填充到 200 维。
    不够 200 维的部分补 0。

    参数：
        filepath: 文件路径
        offset:   样本在文件中的起始字节偏移（已包含头部 8 字节）
        N:        节点数
        E:        边数（分支数）

    返回：
        edge_index: numpy [2, E]
        edge_attr:  numpy [E, 4]
        x:          numpy [N, 200]
        total_cost, total_weight, total_length: float
    """
    with open(filepath, 'rb') as f:
        f.seek(offset)

        # 跳过头部（build_sample_index 已经读过）
        f.read(config.HEADER_BYTES)

        # 读取 edge_index [2, E]
        edge_index_bytes = E * 2 * 4
        edge_index = np.frombuffer(
            f.read(edge_index_bytes), dtype='>i4'
        ).reshape(2, E).astype('<i4')

        # 读取 edge_attr [E, 4]
        edge_attr_bytes = E * config.EDGE_FEAT_DIM * 4
        edge_attr = np.frombuffer(
            f.read(edge_attr_bytes), dtype='>f4'
        ).reshape(E, config.EDGE_FEAT_DIM).astype('<f4')

        # 读取 x [N, 200]
        x_bytes = N * config.NODE_FEAT_DIM * 4
        x = np.frombuffer(
            f.read(x_bytes), dtype='>f4'
        ).reshape(N, config.NODE_FEAT_DIM).astype('<f4')

        # 读取 y: 总成本、总重量、总长度
        total_cost,   = struct.unpack('>f', f.read(4))
        total_weight, = struct.unpack('>f', f.read(4))
        total_length, = struct.unpack('>f', f.read(4))

        return edge_index, edge_attr, x, total_cost, total_weight, total_length


def read_sample(file_list, file_idx, offset, N, E):
    """
    读取样本（用于训练），只返回训练用的 cost 标签。

    返回：
        edge_index, edge_attr, x, cost
    """
    filepath = file_list[file_idx]
    edge_index, edge_attr, x, cost, _, _ = read_sample_from_file(filepath, offset, N, E)
    return edge_index, edge_attr, x, cost


def read_sample_full(file_list, file_idx, offset, N, E):
    """
    读取完整样本（用于统计），返回全部三个标签。

    返回：
        edge_index, edge_attr, x, total_cost, total_length, total_weight
    """
    filepath = file_list[file_idx]
    return read_sample_from_file(filepath, offset, N, E)


def sample_to_tensor(edge_index, edge_attr, x, y):
    """
    将 numpy 数组转换为 PyTorch tensor。

    参数：
        edge_index: numpy [2, E]   (E 可变)
        edge_attr:  numpy [E, 4]
        x:          numpy [N, 200] (N 可变, 200 固定)
        y:          float

    返回：
        对应的 torch tensor
    """
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_t  = torch.tensor(edge_attr,  dtype=torch.float)
    x_t          = torch.tensor(x,          dtype=torch.float)
    y_t          = torch.tensor([y],        dtype=torch.float)
    return edge_index_t, edge_attr_t, x_t, y_t


