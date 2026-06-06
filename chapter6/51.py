from gensim.models import KeyedVectors

# Google Newsで学習済みの単語ベクトルファイル
# 事前に GoogleNews-vectors-negative300.bin.gz をダウンロードして、
# このPythonファイルと同じフォルダに置いておく
file_path = "GoogleNews-vectors-negative300.bin.gz"

# word2vec形式の学習済み単語ベクトルを読み込む
# binary=True は、このファイルがバイナリ形式で保存されているため必要
model = KeyedVectors.load_word2vec_format(file_path, binary=True)

# 比較したい2つの単語を指定する
word1 = "United_States"
word2 = "U.S."

# 2つの単語ベクトルのコサイン類似度を計算する
similarity = model.similarity(word1, word2)

# 結果を表示する
print(f"{word1}と{word2}のコサイン類似度:")
print(similarity)

# 出力
# United_StatesとU.S.のコサイン類似度:
# 0.7310774
