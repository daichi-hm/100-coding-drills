import torch
import torch.nn.functional as F
from itertools import combinations
from transformers import BertTokenizer, BertModel

MODEL_NAME = "bert-base-uncased"

# 類似度を求めたい4つの文
sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

# BERT用のtokenizerを読み込む
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# BERT本体を読み込む
# 84番では[MASK]予測ではなく、各トークンのベクトルを取り出したいのでBertModelを使う
model = BertModel.from_pretrained(MODEL_NAME)

# 今回は学習ではなく、ベクトルを取り出すだけなので評価モードにする
model.eval()

# 複数文をまとめてBERTに入力できる形に変換する
# padding=True:
#   文の長さが違う場合、短い文に[PAD]を追加して長さをそろえる
# return_tensors="pt":
#   PyTorchのTensorとして返す
inputs = tokenizer(
    sentences,
    padding=True,
    return_tensors="pt",
)

# 入力IDを取り出す
input_ids = inputs["input_ids"]

# 確認用：ID列をトークン列に戻して表示
print("tokens:")
for i, ids in enumerate(input_ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"{i}: {tokens}")

# ベクトル抽出だけなので、勾配計算は不要
with torch.no_grad():
    outputs = model(**inputs)

# BERT最終層の各トークンのベクトル
# shape: [文の数, トークン数, 隠れ層の次元数]
last_hidden_state = outputs.last_hidden_state

print("\nlast_hidden_state shape:")
print(last_hidden_state.shape)

# attention_maskは、実トークンなら1、[PAD]なら0
attention_mask = inputs["attention_mask"]

# [CLS], [SEP], [PAD]を平均から除外するためのマスクを作る
# True の位置だけを平均に使う
valid_token_mask = (
    (attention_mask == 1)
    & (input_ids != tokenizer.cls_token_id)
    & (input_ids != tokenizer.sep_token_id)
    & (input_ids != tokenizer.pad_token_id)
)

# valid_token_mask の shape は [文の数, トークン数]
# last_hidden_state の shape は [文の数, トークン数, 768]
# 掛け算できるように、最後に次元を1つ追加する(-1にするとそうなる)
# shape: [文の数, トークン数, 1]
valid_token_mask = valid_token_mask.unsqueeze(-1).float()

# 特殊トークン以外のベクトルだけ残し、
# [CLS], [SEP], [PAD]の位置は0にする
masked_hidden_state = last_hidden_state * valid_token_mask

# トークン方向に足し合わせる
# shape: [文の数, 768]
sum_vectors = masked_hidden_state.sum(dim=1)

# 文ごとに、平均に使ったトークン数を数える
# shape: [文の数, 1]
token_counts = valid_token_mask.sum(dim=1)

# 各文について、トークンベクトルの平均を取る
# shape: [文の数, 768]
mean_vectors = sum_vectors / token_counts

print("\nmean_vectors shape:")
print(mean_vectors.shape)

# すべての文の組み合わせについてコサイン類似度を計算する
print("\ncosine similarities:")
for i, j in combinations(range(len(sentences)), 2):
    similarity = F.cosine_similarity(
        mean_vectors[i],
        mean_vectors[j],
        dim=0,
    )

    print(f"{i} - {j}: {similarity.item():.6f}")
    print(f"  sentence {i}: {sentences[i]}")
    print(f"  sentence {j}: {sentences[j]}")
    print()

#出力
# tokens:
# 0: ['[CLS]', 'the', 'movie', 'was', 'full', 'of', 'fun', '.', '[SEP]']
# 1: ['[CLS]', 'the', 'movie', 'was', 'full', 'of', 'excitement', '.', '[SEP]']
# 2: ['[CLS]', 'the', 'movie', 'was', 'full', 'of', 'crap', '.', '[SEP]']
# 3: ['[CLS]', 'the', 'movie', 'was', 'full', 'of', 'rubbish', '.', '[SEP]']

# last_hidden_state shape:
# torch.Size([4, 9, 768])

# mean_vectors shape:
# torch.Size([4, 768])

# cosine similarities:
# 0 - 1: 0.953455
#   sentence 0: The movie was full of fun.
#   sentence 1: The movie was full of excitement.

# 0 - 2: 0.837856
#   sentence 0: The movie was full of fun.
#   sentence 2: The movie was full of crap.

# 0 - 3: 0.804551
#   sentence 0: The movie was full of fun.
#   sentence 3: The movie was full of rubbish.

# 1 - 2: 0.825631
#   sentence 1: The movie was full of excitement.
#   sentence 2: The movie was full of crap.

# 1 - 3: 0.784942
#   sentence 1: The movie was full of excitement.
#   sentence 3: The movie was full of rubbish.

# 2 - 3: 0.918173
#   sentence 2: The movie was full of crap.
#   sentence 3: The movie was full of rubbish.