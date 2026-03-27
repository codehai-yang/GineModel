import numpy as np
import struct

filepath = r"F:\office\pythonProjects\GINEModel\Samples\Samples_20260319_101851_321"

with open(filepath, 'rb') as f:
    f.read(1688)  # 跳过 edge_index

    ea = np.frombuffer(f.read(3376), dtype='>f4').reshape(211, 4)
    print("=== edge_attr [211,4] ===")
    print(f"第1列非零数: {np.count_nonzero(ea[:, 0])}")
    print(f"第2列非零数: {np.count_nonzero(ea[:, 1])}")
    print(f"第3列非零数: {np.count_nonzero(ea[:, 2])}")
    print(f"第4列非零数: {np.count_nonzero(ea[:, 3])}")  # 你说的全0列？

    x = np.frombuffer(f.read(123200), dtype='>f4').reshape(175, 176)
    print("\n=== x [175,176] ===")
    print(f"前175列非零数: {np.count_nonzero(x[:, :175])}")
    print(f"第176列非零数: {np.count_nonzero(x[:, 175])}")  # 你说的全0列？