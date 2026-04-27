import numpy as np
import torch
import GINEClassifier as gineModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = gineModel.CostModelV2()
model.load_state_dict(torch.load(r'F:\office\pythonProjects\GINEModel\Pt\best_model.pt', map_location=device))
model.to(device)
model.eval()

def predict_single(model, filepath):
    with open(filepath, 'rb') as f:
        # 读取 edge_index [2, 211]
        edge_index = np.frombuffer(f.read(1688), dtype='>i4').reshape(2, 211).copy()
        edge_index = edge_index.astype('<i4')

        # 读取 edge_attr [211, 4]
        edge_attr = np.frombuffer(f.read(3376), dtype='>f4').reshape(211, 4).copy()
        edge_attr = edge_attr.astype('<f4')

        # 读取 x [175, 176]
        x = np.frombuffer(f.read(123200), dtype='>f4').reshape(175, 176).copy()
        x = x.astype('<f4')

    # 注意：Java已经做了标准化，这里不要再调用 normalize_all！

    # 转tensor
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(device)
    edge_attr_t  = torch.tensor(edge_attr,  dtype=torch.float).to(device)
    x_t          = torch.tensor(x,          dtype=torch.float).to(device)

    with torch.no_grad():
        pred = model(x_t, edge_index_t, edge_attr_t)

    return pred.item()

# 使用
pred_cost = predict_single(model, r'F:\office\pythonProjects\GINEModel\javaTest\predict_input.bin')
print(f"预测成本: {pred_cost:.4f}")