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


def training_my_model(
    MODEL,
    TOKENIZER,
    TRAIN_DATA,
    EVAL_DATA,
    output_dir,
    num_epochs=5,
    batch_size=2,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    fp16=True,
):
    # 変数セット
    model = MODEL
    tokenizer = TOKENIZER
    train_dataset = TRAIN_DATA
    eval_dataset = EVAL_DATA

    # 出力ディレクトリ準備
    os.makedirs(output_dir, exist_ok=True)

    # Compute total training steps to set warmup
    total_steps = (
        len(train_dataset) // (batch_size) * num_epochs
    )
    warmup_steps = int(total_steps * warmup_ratio)

    # TrainingArguments にチェックポイント保存設定を追加
    # Define training arguments with checkpointing, scheduling, and mixed precision
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        #evaluation_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=warmup_steps,
        logging_strategy="steps",
        logging_steps=100,
        fp16=fp16,
        max_grad_norm=max_grad_norm,
        report_to=["tensorboard"],
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    # Define compute_metrics for regression (or customize for classification)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # If regression: return MSE and RMSE
        mse = ((predictions - labels) ** 2).mean()
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}

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
