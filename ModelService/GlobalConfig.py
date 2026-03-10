# ============================================================
# 全局配置
# ============================================================
from torch.ao.nn.quantized import Dropout

# 每个样本各字段的维度
NUM_BRANCHES   = 211    # 分支数量（边数）
NUM_NODES      = 175    # 分支点数量（节点数）
EDGE_FEAT_DIM  = 4      # 边特征维度：通断(3维) + 分支长度(1维)
NODE_FEAT_DIM  = 176    #分支点特征维度

# 每个样本的字节数（用于seek定位）
# edge_attr:  211 × 4 × 4字节(float32) = 3376字节
# edge_index: 2 × 211 × 4字节(int32)   = 1688字节
# x:          175 × 176 × 4字节(float32) = 123200字节
# y:          3 × 4字节(float32)        = 12字节  总长度，总成本，总重量
EDGE_ATTR_BYTES  = NUM_BRANCHES * EDGE_FEAT_DIM * 4
EDGE_INDEX_BYTES = 2 * NUM_BRANCHES * 4
X_BYTES          = NUM_NODES * NODE_FEAT_DIM  * 4
Y_BYTES          = 4 * 3
SAMPLE_BYTES     = EDGE_ATTR_BYTES + EDGE_INDEX_BYTES + X_BYTES + Y_BYTES       #一个样本的字节数 128276字节

# 训练超参数
BATCH_SIZE     = 30      # 每个batch的样本数
NUM_EPOCHS     = 200     # 最大训练轮数
LEARNING_RATE  = 0.001   # 学习率
HIDDEN_DIM     = 64      # GINE隐藏层维度
NUM_LAYERS     = 3       # GINE层数
VALIDATE_EVERY = 5    # 每隔多少个batch验证一次
VAL_BATCH_SIZE = 1000    # 每次验证随机抽多少个验证样本
PATIENCE       = 10      # 早停：连续多少次验证无改善就停止
DROUPUT        = 0.5     # Dropout概率,防止过拟合

# 文件路径
SAMPLE_SAVE = 'F:\office\pythonProjects\GINEModel\Samples'                  #所有样本存放目录
TRAIN_FILES  = []  # 训练数据文件列表
MODEL_SAVE   = 'F:\office\pythonProjects\GINEModel\Pt'                      # 模型保存路径
RANDOM_SEED  = 42                                   # 随机种子，保证结果可复现