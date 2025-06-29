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

tokenizer, model = set_model(MODEL_ID, DIR_NAME)
print(model)


## 3. データセットのインポート
DATA_PATH = "datasets/datasets_01.csv"
texts, scores = set_dataset(DATA_PATH)

## 4. 学習前のモデルで予測タスク実行
score = predict(MODEL=model, TOKENIZER=tokenizer, DATA=texts) 
print(score)


## 5. 学習を実行
# データセットの分割
train_dataset, eval_dataset = create_dataset(tokenizer, texts, scores)

# 学習を実施
model = training_my_model(
    MODEL=model, 
    TOKENIZER=tokenizer, 
    TRAIN_DATA=train_dataset, 
    EVAL_DATA=eval_dataset
)

score = predict(MODEL=model, TOKENIZER=tokenizer, DATA=texts) 
print(score)

