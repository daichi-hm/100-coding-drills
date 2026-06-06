from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram
import gc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Google Newsで学習済みの単語ベクトルファイル
file_path = "GoogleNews-vectors-negative300.bin.gz"

# 単語アナロジーの評価データ
# 国名を取り出すために使う
analogy_file = "questions-words.txt"

# word2vec形式の学習済み単語ベクトルを読み込む
print("モデル読み込み開始", flush=True)
model = KeyedVectors.load_word2vec_format(file_path, binary=True)
print("モデル読み込み完了", flush=True)
# 国名を重複なしで保存するための集合
countries = set()

# 国名を取り出す対象セクション
# どちらも「首都と国」の関係を扱うセクション
target_sections = {
    "capital-common-countries",
    "capital-world",
}

# 現在読んでいるセクション名を保存する変数
current_section = ""

# questions-words.txt を読み込む
with open(analogy_file, "r", encoding="utf-8") as f:

    # ファイルを1行ずつ読む
    for line in f:
        # 行末の改行を削除する
        line = line.strip()

        # セクション名の行は ":" で始まる
        if line.startswith(":"):
            # 例: ": capital-common-countries" から
            # "capital-common-countries" だけを取り出す
            current_section = line[2:]
            continue

        # 対象セクション以外は処理しない
        if current_section not in target_sections:
            continue

        # 1行を4つの単語に分ける
        # 例: Athens Greece Baghdad Iraq
        word1, word2, word3, word4 = line.split()

        # capital系セクションでは、
        # 2列目と4列目が国名
        countries.add(word2)
        countries.add(word4)

# 国名と、その国名の単語ベクトルを入れるリスト
country_names = []
country_vectors = []

# 集めた国名から、単語ベクトルが存在するものだけを取り出す
for country in countries:
    if country in model:
        country_names.append(country)
        country_vectors.append(model[country])

# 国名をアルファベット順に並べる
# 結果を毎回同じ順番にし、デンドログラムを見やすくするため
sorted_data = sorted(zip(country_names, country_vectors))

# 並べ替えた結果を、国名リストとベクトルリストに分け直す
country_names = [country for country, vector in sorted_data]
country_vectors = np.array([vector for country, vector in sorted_data])

print("国名リストとベクトル分け完了", flush=True)

# 描画前に巨大な学習済みモデルを解放する
del model
gc.collect()
print("モデル解放完了", flush=True)

# Ward法による階層型クラスタリングを実行する
# method="ward" が Ward法を指定している部分
linkage_result = linkage(country_vectors, method="ward")

print("クラスタリング実行", flush=True)

# デンドログラムを描画するための図のサイズを指定する
fig, ax = plt.subplots(figsize=(18, 8))

print("デンドログラム描画開始", flush=True)

# デンドログラムを描画する
# labels に国名を渡すことで、葉の部分に国名が表示される
dendrogram(
    linkage_result,
    labels=country_names,
    leaf_rotation=90,
    leaf_font_size=8,
    ax=ax,
)

# 保存先を明示する
output_file = Path(__file__).with_name("answer_58_dendrogram.png")

ax.set_title("Hierarchical Clustering of Country Vectors")
ax.set_xlabel("Country")
ax.set_ylabel("Distance")
fig.tight_layout()

print("画像を保存します...", flush=True)

fig.savefig(output_file, dpi=200)
plt.close(fig)

print("保存処理が終わりました。")

# 結果は"answer_58_dendrogram.png"に保存されている
