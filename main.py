## 1. ライブラリをインポート
# 標準ライブラリ
import torch, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
# 外部ファイル
from download_model import set_model
from load_datasets import set_dataset, create_dataset
from predict_before_learning import predict
from training_model import training_my_model

## 2. AIモデルのインポート
MODEL_ID = "megagonlabs/roberta-long-japanese"
DIR_NAME = "models"

# ディレクトリ構成
BASE_MODEL_DIR = os.path.join(DIR_NAME, MODEL_ID)
FINE_TUNE_DIR = os.path.join(BASE_MODEL_DIR, "fine_tuned")
os.makedirs(FINE_TUNE_DIR, exist_ok=True)

print("2. モデルのインストール")
# 前回のチェックポイントがあればロード
checkpoints = [d for d in os.listdir(FINE_TUNE_DIR) if d.startswith("checkpoint-")]
if checkpoints:
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    ckpt_path = os.path.join(FINE_TUNE_DIR, latest)
    print(f"Resuming from checkpoint: {ckpt_path}")
    tokenizer, model = set_model(MODEL_ID, DIR_NAME)
    # 最新チェックポイントからモデルロード
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path, num_labels=1)
else:
    tokenizer, model = set_model(MODEL_ID, DIR_NAME)

#print(model)

## 3. データセットのインポート
print("3. データセットのインポート")
#DATA_PATH = "datasets/datasets_01.csv"
DATA_PATH = "datasets/output.csv"
texts, scores = set_dataset(DATA_PATH)

## 4. 学習前のモデルで予測タスク実行
print("4. 学習前のモデルで予測タスク実行")
score = predict(MODEL=model, TOKENIZER=tokenizer, DATA=texts) 
print(score)

# 学習データセット作成
print("5-1. 学習用データセットの準備")
train_dataset, eval_dataset = create_dataset(tokenizer, texts, scores)

# ファインチューニング実行
print("5-2. 学習開始")
model = training_my_model(
    MODEL=model,
    TOKENIZER=tokenizer,
    TRAIN_DATA=train_dataset,
    EVAL_DATA=eval_dataset,
    output_dir=FINE_TUNE_DIR,
    num_epochs=100
)

# 最新モデルを 'latest' フォルダに保存（上書き）
latest_dir = os.path.join(FINE_TUNE_DIR, "latest")
if os.path.exists(latest_dir):
    import shutil; shutil.rmtree(latest_dir)
model.save_pretrained(latest_dir)
tokenizer.save_pretrained(latest_dir)

score = predict(MODEL=model, TOKENIZER=tokenizer, DATA=texts) 
print(score)

