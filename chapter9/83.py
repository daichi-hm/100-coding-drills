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
# 81, 82番では[MASK]予測用に BertForMaskedLM を使ったが、
# 83番では文ベクトルを取り出したいので BertModel を使う
model = BertModel.from_pretrained(MODEL_NAME)

# 今回は学習ではなく、ベクトルを取り出すだけなので評価モードにする
model.eval()

# 4つの文をまとめてBERTに入力できる形に変換する
# padding=True:
#   文の長さが違っても、同じ長さになるように[PAD]を追加する
# return_tensors="pt":
#   PyTorchのTensorとして返す
inputs = tokenizer(
    sentences,
    padding=True,
    return_tensors="pt",
)

# 予測・ベクトル抽出だけなので、勾配計算は不要
with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state には、
# 最終層における各トークンの埋め込みベクトルが入っている
# shapeは [文の数, トークン数, 隠れ層の次元数]
last_hidden_state = outputs.last_hidden_state

# [CLS]トークンは各文の先頭、つまり0番目の位置にある
# そのため、各文の[CLS]ベクトルは last_hidden_state[:, 0, :] で取り出せる
# shapeは [文の数, 隠れ層の次元数]
cls_vectors = last_hidden_state[:, 0, :]

# すべての文の組み合わせについてコサイン類似度を計算する
for i, j in combinations(range(len(sentences)), 2):
    # cls_vectors[i] と cls_vectors[j] は、それぞれ文i, 文jの[CLS]ベクトル
    similarity = F.cosine_similarity(
        cls_vectors[i],
        cls_vectors[j],
        dim=0,
    )

    print(f"{i} - {j}: {similarity.item():.6f}")
    print(f"  sentence {i}: {sentences[i]}")
    print(f"  sentence {j}: {sentences[j]}")
    print()

#出力
# 0 - 1: 0.988061
#   sentence 0: The movie was full of fun.
#   sentence 1: The movie was full of excitement.

# 0 - 2: 0.955766
#   sentence 0: The movie was full of fun.
#   sentence 2: The movie was full of crap.

# 0 - 3: 0.947533
#   sentence 0: The movie was full of fun.
#   sentence 3: The movie was full of rubbish.

# 1 - 2: 0.954127
#   sentence 1: The movie was full of excitement.
#   sentence 2: The movie was full of crap.

# 1 - 3: 0.948664
#   sentence 1: The movie was full of excitement.
#   sentence 3: The movie was full of rubbish.

# 2 - 3: 0.980693
#   sentence 2: The movie was full of crap.
#   sentence 3: The movie was full of rubbish.