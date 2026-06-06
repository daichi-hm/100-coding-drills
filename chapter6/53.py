from gensim.models import KeyedVectors

# Google Newsで学習済みの単語ベクトルファイル
# 事前に GoogleNews-vectors-negative300.bin.gz をダウンロードして、
# このPythonファイルと同じフォルダに置いておく
file_path = "GoogleNews-vectors-negative300.bin.gz"

# word2vec形式の学習済み単語ベクトルを読み込む
# binary=True は、このファイルがバイナリ形式で保存されているため必要
model = KeyedVectors.load_word2vec_format(file_path, binary=True)

# Spain - Madria + Athens　に近い単語を上位10件取得する
# positiveには「足す単語」を指定する
# negativeには「引く単語」を指定する
similar_words = model.most_similar(
    positive=["Spain", "Athens"], negative=["Madrid"], topn=10
)

# 結果を表示する
for similar_word, similarity in similar_words:
    print(f"{similar_word}:{similarity}")

# 出力
# Greece:0.6898481845855713
# Aristeidis_Grigoriadis:0.560684859752655
# Ioannis_Drymonakos:0.5552908778190613
# Greeks:0.5450685620307922
# Ioannis_Christou:0.5400861501693726
# Hrysopiyi_Devetzi:0.5248444676399231
# Heraklio:0.5207760334014893
# Athens_Greece:0.5168810486793518
# Lithuania:0.5166866183280945
# Iraklion:0.5146791934967041
