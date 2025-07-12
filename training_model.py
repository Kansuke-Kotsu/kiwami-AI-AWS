# training_model.py
import os
from pathlib import Path
import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

# カスタム collate_fn の定義

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch], dim=0)
    attention_mask = torch.stack([item[1] for item in batch], dim=0)
    labels = torch.stack([item[2] for item in batch], dim=0)
    return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels
            }


def training_my_model(MODEL, TOKENIZER, TRAIN_DATA, EVAL_DATA, output_dir, num_epochs=5):
    # 変数セット
    model = MODEL
    tokenizer = TOKENIZER
    train_dataset = TRAIN_DATA
    eval_dataset = EVAL_DATA

    # 出力ディレクトリ準備
    os.makedirs(output_dir, exist_ok=True)

    # TrainingArguments にチェックポイント保存設定を追加
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=100,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    # 既存チェックポイントからリジューム
    checkpoints = [p for p in Path(output_dir).glob("checkpoint-*") if p.is_dir()]
    if checkpoints:
        latest_ckpt = sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))[-1]
        trainer.train(resume_from_checkpoint=str(latest_ckpt))
    else:
        trainer.train()

    return trainer.model
