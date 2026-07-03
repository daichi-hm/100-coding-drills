from transformers import BertTokenizer

# 今回はGoogleのBERT base uncasedを使う
MODEL_NAME = "bert-base-uncased"

# トークン化したい文
text = "The movie was full of incomprehensibilities."

# BERT用のtokenizerを読み込む
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 文をトークン列に分解する
tokens = tokenizer.tokenize(text)

# 結果を表示する
print(tokens)

#出力
# ['the', 'movie', 'was', 'full', 'of', 'inc', '##omp', '##re', '##hen', '##si', '##bilities', '.']