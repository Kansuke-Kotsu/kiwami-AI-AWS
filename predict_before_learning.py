########################
# 学習前のモデルでスコアリング
########################

## 途中でトークンの切り詰めをおこなっているので注意
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def predict(MODEL, TOKENIZER, DATA):
    # 0) 変数をセット
    model_pre = MODEL
    tokenizer = TOKENIZER
    texts = DATA

    # 1) モデル設定を書き換える
    new_vocab_size = len(tokenizer)
    model_pre.config.vocab_size = new_vocab_size

    # 2) 埋め込み行列をリサイズ
    model_pre.resize_token_embeddings(new_vocab_size)

    # 以降は同じく eval モード→トークナイズ→推論
    model_pre.eval()


    # トークナイズ
    enc_pre = tokenizer(
        texts[0],
        padding="max_length",
        truncation=True,
        max_length=model_pre.config.max_position_embeddings,
        return_tensors="pt"
    )

    # 埋め込み行列の shape を取得
    emb_weight = model_pre.get_input_embeddings().weight
    print("embedding rows:", emb_weight.size(0))     # 本当に何行ある？
    print("config.vocab_size:", model_pre.config.vocab_size)
    print("tokenizer.vocab size:", len(tokenizer))
    #print("max token ID in input:", ids.max().item())

    # モデルの最大長
    model_max = model_pre.config.max_position_embeddings  # 例：1282

    # 実用的な上限（512 or 1024など）を選ぶ
    SAFE_LEN = min(512, model_max)

    # トークナイズ
    enc_pre = tokenizer(
        texts[0],
        padding="max_length",
        truncation=True,
        max_length=SAFE_LEN,
        return_tensors="pt"
    )

    ids = enc_pre["input_ids"][0]
    print("max token ID now:", ids.max().item())

    # 推論
    with torch.no_grad():
        out = model_pre(**enc_pre)

    return out.logits