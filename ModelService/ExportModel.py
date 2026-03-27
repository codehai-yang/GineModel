# 导出ONNX模型
import torch
import GINEClassifier as gineModel
import GlobalConfig as config
import onnxruntime as ort
import numpy as np

model = gineModel.CostModelV2()
model.load_state_dict(torch.load(config.MODEL_SAVE))
model.eval()

# dummy输入
dummy_x          = torch.randn(175, 176)
dummy_edge_index = torch.randint(0, 175, (2, 211)).long()
dummy_edge_attr  = torch.randn(211, 4)


def exportModel():
    try:
        torch.onnx.export(
            model,
            (dummy_x, dummy_edge_index, dummy_edge_attr),
            'cost_model.onnx',
            input_names   = ['x', 'edge_index', 'edge_attr'],
            output_names  = ['predicted_cost'],
            opset_version = 12
        )
        print('导出成功')
    except Exception as e:
        print(f'导出失败：{e}')
        print('建议使用 HTTP 接口方案')


def testModel():
    sess = ort.InferenceSession('cost_model.onnx')

    # 用和导出时一样的 dummy 数据
    result = sess.run(['predicted_cost'], {
        'x':          np.random.randn(175, 176).astype(np.float32),
        'edge_index': np.random.randint(0, 175, (2, 211)).astype(np.int64),
        'edge_attr':  np.random.randn(211, 4).astype(np.float32),
    })
    print(result)
# 导出
if __name__ == '__main__':
    # exportModel()
    testModel()

