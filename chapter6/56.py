from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import csv

# Google Newsで学習済みの単語ベクトルファイル
file_path = "GoogleNews-vectors-negative300.bin.gz"

# WordSimilarity-353の評価データ
# combined.csv をこのPythonファイルと同じフォルダに置いておく
similarity_file = "combined.csv"

# word2vec形式の学習済み単語ベクトルを読み込む
model = KeyedVectors.load_word2vec_format(file_path, binary=True)

# 人間が付けた類似度スコアを入れるリスト
human_scores = []

# 単語ベクトルで計算した類似度スコアを入れるリスト
model_scores = []

# WordSimilarity-353のCSVファイルを開く
with open(similarity_file, "r", encoding="utf-8") as f:
    # CSVファイルを辞書形式で読み込む
    reader = csv.DictReader(f)

    # CSVファイルを1行ずつ読む
    for row in reader:
        # 1つ目の単語
        word1 = row["Word 1"]

        # 2つ目の単語
        word2 = row["Word 2"]

        # 人間が付けた類似度スコア
        # CSVでは文字列として読まれるので、floatに変換する
        human_score = float(row["Human (mean)"])

        # どちらかの単語がモデルに存在しない場合は、その行を飛ばす
        if word1 not in model or word2 not in model:
            continue

        # 単語ベクトルによるコサイン類似度を計算する
        model_score = model.similarity(word1, word2)

        # 人間のスコアとモデルのスコアを、それぞれリストに追加する
        human_scores.append(human_score)
        model_scores.append(model_score)

# 人間スコアのランキングと、モデルスコアのランキングのスピアマン相関係数を計算する
correlation, p_value = spearmanr(human_scores, model_scores)

# 結果を表示する
print("評価に使った単語ペア数:", len(human_scores))
print("スピアマン相関係数:", correlation)
print("p値:", p_value)

# 出力
# 評価に使った単語ペア数: 353
# スピアマン相関係数: 0.7000166486272194
# p値: 2.86866666051422e-53
