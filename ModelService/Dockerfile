FROM python:3.9-slim
WORKDIR /app

# 只安装依赖
# 第一步：先安装 PyTorch（必须从官方源）
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# 第二步：复制并安装其他依赖（可以使用清华镜像）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

# 固定入口，CMD 只提供方便的默认值（都可被覆盖）
ENTRYPOINT ["python", "train_gine.py"]
CMD ["--epochs", "300", "--lr", "0.001", "--batch_size", "128"]