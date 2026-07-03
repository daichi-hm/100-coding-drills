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

# input_idsの中から[MASK]の位置を探す
# tokenizer.mask_token_idは[MASK]に対応するID
#0番目の情報 → 文番号、1番目の情報 → トークン位置
mask_position = torch.where(input_ids == tokenizer.mask_token_id)[1]

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

# スコアが最も高いトークンIDを求める
predicted_token_id = torch.argmax(mask_logits, dim=-1).item()

# トークンIDを実際の文字列トークンに戻す
predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)

# 結果を表示する
print(predicted_token)

#出力
# fun