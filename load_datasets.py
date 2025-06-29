import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch

def set_dataset(DATA_PATH):
    # 変数をセット
    data_path = DATA_PATH

    # CSV の読み込み（BOM 対策＆空白除去）
    df = pd.read_csv(
        data_path,
        encoding="utf-8-sig",  # BOM を自動除去
    )
    df.columns = df.columns.str.strip()  # 列名の前後空白を除去
    #print("columns:", df.columns.tolist())

    # テキストとスコアをリスト化
    texts = df["text"].astype(str).tolist()
    scores = df["score"].astype(float).tolist()

    return texts, scores

def create_dataset(TOKENIZER, DATAS, SCORES):
    # 変数をセット
    texts = DATAS
    scores = SCORES
    tokenizer = TOKENIZER
    
    # 設定
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, scores, test_size=0.2, random_state=42
    )

    # トークナイズ
    max_len = 1024   # モデルがサポートする最大長を取得(今後要改善)
    enc_train = tokenizer(train_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc_eval  = tokenizer(eval_texts,  padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    # Dataset生成（回帰なので labels を (N,1) に）
    train_dataset = TensorDataset(
        enc_train["input_ids"],
        enc_train["attention_mask"],
        torch.tensor(train_labels, dtype=torch.float).unsqueeze(1)
    )
    eval_dataset = TensorDataset(
        enc_eval["input_ids"],
        enc_eval["attention_mask"],
        torch.tensor(eval_labels,  dtype=torch.float).unsqueeze(1)
    )

    return train_dataset, eval_dataset