from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# ——————————————————————
# 1. トークナイザのロード
# ——————————————————————

MODEL_ID = "megagonlabs/roberta-long-japanese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1  # 回帰用に1出力
)

print(model)