FROM python:3.9-slim
WORKDIR /app

# 只安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 固定入口，CMD 只提供方便的默认值（都可被覆盖）
ENTRYPOINT ["python", "train_gine.py"]
CMD ["--epochs", "300", "--lr", "0.001", "--batch_size", "128"]