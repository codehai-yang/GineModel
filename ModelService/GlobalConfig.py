# ============================================================
# 全局配置
# ============================================================
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_INPUT_DIR = os.environ.get('IWORK_DATA_IN1', '').rstrip('/')
SD_ID_AI_DIR = 'SD_ID_AI'
DATA_OUTPUT_DIR = os.environ.get('IWORK_DATA_OUT', '').rstrip('/')


# 每个样本各字段的维度（基础维度，实际样本 N/E 可变）
EDGE_FEAT_DIM  = 4      # 边特征维度：通断(3维) + 分支长度(1维)
NODE_FEAT_DIM  = 200    # 分支点特征维度（填充到 200 维，不够补 0）
Y_FEAT_COUNT   = 3      # 标签数量：总成本、总长度、总重量

# 样本头部信息
HEADER_ITEMS    = 2         # N 和 E 两个整数
HEADER_BYTES    = HEADER_ITEMS * 4   # 8 字节

# 每个样本的字节数动态计算（N 和 E 可变）
# 公式：HEADER_BYTES + (E * 2 * 4 + E * EDGE_FEAT_DIM * 4 + N * NODE_FEAT_DIM * 4 + Y_FEAT_COUNT * 4)
#     = 8 + (E * 6 + N * 200 + 3) * 4
#     = 8 + E * 24 + N * 800 + 12
#     = 20 + E * 24 + N * 800

# 训练超参数
BATCH_SIZE     = 128      # 每个batch的样本数
NUM_EPOCHS     = 300     # 最大训练轮数
LEARNING_RATE  = 0.001   # 学习率
HIDDEN_DIM     = 64      # GINE隐藏层维度
NUM_LAYERS     = 3       # GINE层数
VALIDATE_EVERY = 125    # 每隔多少个batch验证一次
VAL_BATCH_SIZE = 1000    # 每次验证随机抽多少个验证样本
PATIENCE       = 20      # 早停：连续多少次验证无改善就停止
# DROUPUT        = 0.5     # Dropout概率,防止过拟合

# 文件路径
NORMALIZATION_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'normalization_params.json')
TRAIN_FILES  = []  # 训练数据文件列表
RANDOM_SEED  = 42                                   # 随机种子，保证结果可复现

path1 = os.path.join(DATA_INPUT_DIR, SD_ID_AI_DIR)
path2 = os.path.join(DATA_INPUT_DIR, 'TrainData', SD_ID_AI_DIR)

if os.path.exists(path1) and path1 != DATA_INPUT_DIR:
    SAMPLE_SAVE = path1
elif os.path.exists(path2) and path2 != DATA_INPUT_DIR:
    SAMPLE_SAVE = path2
else:
    SAMPLE_SAVE = DATA_INPUT_DIR

if DATA_OUTPUT_DIR:
    MODEL_SAVE = os.path.join(DATA_OUTPUT_DIR, 'pt', 'best_model.pt')
    LOG_DIR = os.path.join(DATA_OUTPUT_DIR, 'logs')
else:
    # MODEL_SAVE = '/app/pt/best_model.pt'
    MODEL_SAVE = r'F:\office\pythonProjects\GINEModel\Pt\best_model.pt'
    LOG_DIR = '/app/logs'

os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)