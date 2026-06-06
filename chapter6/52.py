from gensim.models import KeyedVectors

# Google Newsで学習済みの単語ベクトルファイル
# 事前に GoogleNews-vectors-negative300.bin.gz をダウンロードして、
# このPythonファイルと同じフォルダに置いておく
file_path = "GoogleNews-vectors-negative300.bin.gz"

# word2vec形式の学習済み単語ベクトルを読み込む
# binary=True は、このファイルがバイナリ形式で保存されているため必要
model = KeyedVectors.load_word2vec_format(file_path, binary=True)

# 類似度を調べたい単語を指定
word = "United_States"

# wordとコサイン類似度が高い単語を上位10件取得する
# most_similarは指定した単語に意味的に近い単語を探す関数
similar_words = model.most_similar(word, topn=10)

# 結果を表示する
for similar_word, similarity in similar_words:
    print(f"{similar_word}:{similarity}")

# 出力
# Unites_States:0.7877249121665955
# Untied_States:0.7541369795799255
# United_Sates:0.7400726079940796
# U.S.:0.7310774326324463
# theUnited_States:0.6404393315315247
# America:0.6178411841392517
# UnitedStates:0.6167312860488892
# Europe:0.6132988333702087
# countries:0.6044804453849792
# Canada:0.6019068956375122
