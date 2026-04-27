FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "train_gine.py"]
CMD [
  "--data_dir", "/app/Samples",
  "--model_save", "/app/Pt/best_model.pt",
  "--batch_size", "128",
  "--epochs", "300",
  "--lr", "0.001",
  "--weight_decay", "0.01",
  "--patience", "20",
  "--validate_every", "125",
  "--val_batch_size", "1000",
  "--hidden_dim", "64",
  "--num_layers", "3",
  "--num_workers_train", "0",
  "--num_workers_eval", "0",
  "--seed", "42",
  "--log_dir", "/app/runs/experiment_name"
]