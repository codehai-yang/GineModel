FROM python:3.9-slim
WORKDIR /app

# 第一步：安装 PyTorch（使用官方索引）
RUN pip install --no-cache-dir \
    torch==2.5.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 第二步：安装 torch_geometric（从 PyPI 直接安装）
# 根据 PyG 官方文档，2.3 及以上版本支持此方法
RUN pip install --no-cache-dir torch_geometric==2.6.1

# 第三步：安装其他依赖（使用清华镜像）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn

COPY . .
ENTRYPOINT ["python", "train_gine.py"]
CMD ["--epochs", "300", "--lr", "0.001", "--batch_size", "128"]