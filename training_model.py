import torch
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


# カスタム collate_fn の定義
#  train_dataset, eval_dataset が TensorDataset の場合、
#  各要素は (input_ids, attention_mask, label) のタプルなので、
#  バッチをまとめて dict に変換します。
def collate_fn(batch):
    input_ids     = torch.stack([item[0] for item in batch], dim=0)
    attention_mask = torch.stack([item[1] for item in batch], dim=0)
    labels        = torch.stack([item[2] for item in batch], dim=0)
    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

def training_my_model(MODEL, TOKENIZER, TRAIN_DATA, EVAL_DATA):
    # 変数をセット
    model = MODEL
    tokenizer = TOKENIZER
    train_dataset = TRAIN_DATA
    eval_dataset = EVAL_DATA
    
    # ファインチューニング実行
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=2e-5,
        eval_strategy="epoch",
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        #data_collator=data_collator,
        data_collator=collate_fn,   # ← ここをカスタム collate_fn に
        #compute_metrics=compute_metrics,   # もしメトリクスを見たいなら
        # tokenizer=tokenizer              # v5.0.0 以降は deprecated なので外してOK
    )

    # 学習実行
    trainer.train()

    return model