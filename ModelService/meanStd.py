import os
import sys
import json
import numpy as np

# 确保能导入项目中的模块
sys.path.append(os.path.dirname(__file__))

import LoadSample as loadSample
import GlobalConfig as config

def compute_stats(file_list, indices):
    """
    基于给定的索引列表计算统计量（全量计算）。
    """
    print(f"正在处理全部 {len(indices)} 个训练样本...")

    all_branch_lengths = []
    all_prices = []
    all_wet_costs = []
    all_total_cost = []
    all_total_length = []
    all_total_weight = []

    for idx, (file_idx, sample_idx) in enumerate(indices):
        try:
            # 读取完整样本，包含三个标签
            edge_index, edge_attr, x, total_cost, total_length, total_weight = \
                loadSample.read_sample_full(file_list, file_idx, sample_idx)

            # 1. 分支长度 (edge_attr 第4列)
            all_branch_lengths.append(edge_attr[:, 3])

            # 2. 回路单价 (x 前175列的非零值)
            price_matrix = x[:, :175]
            nonzero_prices = price_matrix[price_matrix != 0]
            if len(nonzero_prices) > 0:
                all_prices.append(nonzero_prices)

            # 3. 湿区成本 (x 第176列的非零值)
            wet_col = x[:, 175]
            nonzero_wet = wet_col[wet_col != 0]
            if len(nonzero_wet) > 0:
                all_wet_costs.append(nonzero_wet)

            # 4. 样本标签：总成本、总长度、总重量（每个样本一个值）
            all_total_cost.append(total_cost)
            all_total_length.append(total_length)
            all_total_weight.append(total_weight)

            if (idx + 1) % 1000 == 0:
                print(f"  已处理 {idx + 1} / {len(indices)} 个样本...")

        except Exception as e:
            print(f"读取样本 {file_idx}-{sample_idx} 时出错: {e}")
            continue

    if not all_branch_lengths:
        raise ValueError("未能读取到任何有效样本数据！")

    # 合并所有数据
    branch_lengths = np.concatenate(all_branch_lengths)
    prices = np.concatenate(all_prices) if all_prices else np.array([0.0])
    wet_costs = np.concatenate(all_wet_costs) if all_wet_costs else np.array([0.0])

    # 标签数据
    total_costs = np.array(all_total_cost)
    total_lengths = np.array(all_total_length)
    total_weights = np.array(all_total_weight)

    stats = {
        # 特征归一化参数
        'branch_length_mean': float(branch_lengths.mean()),
        'branch_length_std':  float(branch_lengths.std()),
        'price_mean':         float(prices.mean()),
        'price_std':          float(prices.std()),
        'wet_cost_mean':      float(wet_costs.mean()),
        'wet_cost_std':       float(wet_costs.std()),
        # 标签归一化参数
        'total_cost_mean':    float(total_costs.mean()),
        'total_cost_std':     float(total_costs.std()),
        'total_length_mean':  float(total_lengths.mean()),
        'total_length_std':   float(total_lengths.std()),
        'total_weight_mean':  float(total_weights.mean()),
        'total_weight_std':   float(total_weights.std()),
    }

    return stats

def main():
    # 配置路径
    data_dir = r'F:\office\pythonProjects\GINEModel\Samples'
    output_path = os.path.join(r'F:\office\pythonProjects\GINEModel\Pt', 'normalization_params2.json')

    print("=" * 50)
    print("开始计算全局归一化参数（全量训练集）")
    print("=" * 50)

    # 1. 建立索引
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_indices, file_list = loadSample.build_global_indices(data_dir)

    # 2. 划分数据集（只取训练集索引）
    print("\n划分数据集以获取训练集索引...")
    train_indices, _, _ = loadSample.split_indices(all_indices)

    # 3. 计算统计量（传入所有训练集索引）
    print("\n开始统计分析...")
    stats = compute_stats(file_list, train_indices)

    # 4. 打印结果
    print("\n" + "=" * 50)
    print("计算完成！统计结果如下：")
    print("=" * 50)
    print("\n[特征归一化参数]")
    feat_keys = ['branch_length_mean', 'branch_length_std', 'price_mean', 'price_std',
                 'wet_cost_mean', 'wet_cost_std']
    for key in feat_keys:
        if key in stats:
            print(f"  {key}: {stats[key]:.6f}")

    print("\n[标签归一化参数]")
    label_keys = ['total_cost_mean', 'total_cost_std', 'total_length_mean', 'total_length_std',
                  'total_weight_mean', 'total_weight_std']
    for key in label_keys:
        if key in stats:
            print(f"  {key}: {stats[key]:.6f}")

    # 5. 保存文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n参数已保存到: {output_path}")
    print("请在 Normalize.py 中使用 load_normalization_params() 加载此文件。")

if __name__ == '__main__':
    main()
