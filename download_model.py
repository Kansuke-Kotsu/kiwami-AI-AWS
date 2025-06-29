from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# ——————————————————————
# モデル設定
# ——————————————————————
MODEL_ID = "megagonlabs/roberta-long-japanese"
CACHE_DIR = "models/"+MODEL_ID

# キャッシュ先ディレクトリがなければ作成
os.makedirs(CACHE_DIR, exist_ok=True)

# ——————————————————————
# 1. トークナイザ & モデルのロード or キャッシュ保存
# ——————————————————————
config_path = os.path.join(CACHE_DIR, "config.json")
if not os.path.exists(config_path):
    print("モデルをダウンロード中…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1  # 回帰用に1出力
    )
    # ローカルに保存
    tokenizer.save_pretrained(CACHE_DIR)
    model.save_pretrained(CACHE_DIR)
    print("ダウンロード完了 → キャッシュに保存しました。")
else:
    print("ローカルキャッシュからモデルをロード中…")
    tokenizer = AutoTokenizer.from_pretrained(CACHE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        CACHE_DIR,
        num_labels=1
    )
    print("ロード完了。")

print(model)
