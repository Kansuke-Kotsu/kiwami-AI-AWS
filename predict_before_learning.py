from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict(MODEL, TOKENIZER, texts, labels):
    """
    MODEL     : 学習前のモデルオブジェクト
    TOKENIZER : 対応するトークナイザー
    texts     : テキストのリスト
    labels    : texts に対応するラベルのリスト
    """
    model = MODEL
    tokenizer = TOKENIZER

    # 語彙数リサイズ
    new_vocab_size = len(tokenizer)
    model.config.vocab_size = new_vocab_size
    model.resize_token_embeddings(new_vocab_size)
    model.eval()

    SAFE_LEN = min(512, model.config.max_position_embeddings)

    # 先頭データだけを使う
    text = texts[0]
    true_label = labels[0]

    # トークナイズ
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=SAFE_LEN,
        return_tensors="pt"
    )

    # 推論
    with torch.no_grad():
        out = model(**enc)

    logits = out.logits.squeeze().cpu()
    # 分類タスクなら argmax、回帰タスクならそのまま
    # --- 分類の場合 ---
    # pred = torch.argmax(logits).item()
    # --- 回帰の場合 ---
    pred = logits.item()

    # 結果表示
    print(f"text: {text[:50]}{'...' if len(text)>50 else ''}")
    print(f"predicted: {pred:.4f}    true: {true_label}")

    return pred

# 使い方例
# MODEL = AutoModelForSequenceClassification.from_pretrained("...")
# TOKENIZER = AutoTokenizer.from_pretrained("...")
# texts = ["サンプル文1", "サンプル文2", ...]
# labels = [0.75, 0.23, ...]
# pred0 = predict_once(MODEL, TOKENIZER, texts, labels)
