from torch_geometric.data import Dataset, Data
import LoadSample as loadSample
import Normalize as nz


class GraphDataset(Dataset):
    """
    将 (file_idx, sample_idx, offset, N, E) 索引列表封装为 PyG Dataset。
    DataLoader 会用多个 worker 进程并行调用 get()，
    彻底解决 CPU 串行读取/归一化拖慢 GPU 的问题。
    """

    def __init__(self, file_list, indices):
        """
        参数：
            file_list : 数据文件路径列表（来自 loadSample.build_sample_index）
            indices   : [(file_idx, sample_idx, offset, N, E), ...] 样本索引列表
        """
        super().__init__()
        self.file_list = file_list
        self.sample_indices   = indices

    def len(self):
        return len(self.sample_indices)

    def get(self, idx):
        """
        被 DataLoader worker 调用，返回单个图的 Data 对象。
        归一化在这里做，完全在 CPU worker 里并行执行。
        """
        file_idx, sample_idx, offset, N, E = self.sample_indices[idx]

        # 读取原始数据（按偏移量读取，N/E 可变）
        edge_index, edge_attr, x, y = loadSample.read_sample(
            self.file_list, file_idx, offset, N, E
        )

        # 归一化（在 worker 进程里并行，不占主进程/GPU时间）
        # 传入 N：标准化回路单价前 N 列，湿区第 N 列
        edge_attr, x, y = nz.normalize_all(edge_attr, x, y, N)

        # 转为 tensor（维度可变）
        edge_index_t, edge_attr_t, x_t, y_t = loadSample.sample_to_tensor(
            edge_index, edge_attr, x, y
        )

        # y 统一为 shape [1]，方便 Batch 后 cat 成 [B]
        if y_t.dim() == 0:
            y_t = y_t.unsqueeze(0)

        return Data(
            x          = x_t,           # [N, 200]
            edge_index = edge_index_t,  # [2, E]
            edge_attr  = edge_attr_t,   # [E, 4]
            y          = y_t            # [1]
        )
