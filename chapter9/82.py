import torch
from transformers import BertTokenizer, BertForMaskedLM

MODEL_NAME = "bert-base-uncased"

# [MASK]を含む文
text = "The movie was full of [MASK]."

# BERT用のtokenizerを読み込む
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# マスク単語予測用のBERTモデルを読み込む
# BertForMaskedLMは、BERT本体の上に「各トークンの語彙スコアを出す層」が付いたモデル
model = BertForMaskedLM.from_pretrained(MODEL_NAME)

# 今回は学習ではなく予測だけなので、評価モードにする
model.eval()

# 文をBERTに入力できる形に変換する
# return_tensors="pt"を指定すると、PyTorchのTensorとして返される
inputs = tokenizer(text, return_tensors="pt")

# 入力IDを取り出す
# input_idsの中には、[CLS], the, movie, ..., [MASK], ., [SEP]のIDが入っている
input_ids = inputs["input_ids"]

# 入力IDをトークン列に戻して確認する
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("tokens:")
print(tokens)

# input_idsの中から[MASK]の位置を探す
# tokenizer.mask_token_idは[MASK]に対応するID
#0番目の情報 → 文番号、1番目の情報 → トークン位置
mask_position = torch.where(input_ids == tokenizer.mask_token_id)[1].item()

# 予測だけなので、勾配計算は不要
# torch.no_grad()を使うと、余計な計算を省ける
with torch.no_grad():
    outputs = model(**inputs)

# outputs.logitsには、各位置ごとの各語彙トークンのスコアが入っている
# shapeは [バッチサイズ, トークン数, 語彙数]
logits = outputs.logits

# [MASK]の位置に対する語彙スコアだけを取り出す
#logits[文番号, トークン位置, 語彙候補]
mask_logits = logits[0, mask_position, :]

# logitsは確率ではないので、softmaxで確率に変換する
# dim=0 は「語彙方向」、つまり全語彙候補に対してsoftmaxするという意味
probabilities = torch.softmax(mask_logits, dim=0)

# 確率が高い上位10個を取り出す
# topk_probs: 上位10個の確率
# topk_ids: 上位10個のトークンID
topk_probs, topk_ids = torch.topk(probabilities, k=10)

# 上位10個のトークンと確率を表示する
print("\ntop 10 predictions:")
for token_id, prob in zip(topk_ids, topk_probs):
    token = tokenizer.convert_ids_to_tokens(token_id.item())
    print(f"{token}\t{prob.item():.6f}")

# #出力
# tokens:
# ['[CLS]', 'the', 'movie', 'was', 'full', 'of', '[MASK]', '.', '[SEP]']

# top 10 predictions:
# fun     0.107119
# surprises       0.066345
# drama   0.044684
# stars   0.027217
# laughs  0.025413
# action  0.019517
# excitement      0.019038
# people  0.018290
# tension 0.015031
# music   0.014646