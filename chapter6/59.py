from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import gc
import matplotlib

# 画面表示ではなく、画像ファイル保存用の描画モードにする
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

        # capital系セクションでは、2列目と4列目が国名
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
# 結果を毎回同じ順番にし、図を見やすくするため
sorted_data = sorted(zip(country_names, country_vectors))

# 並べ替えた結果を、国名リストとベクトル配列に分け直す
country_names = [country for country, vector in sorted_data]
country_vectors = np.array([vector for country, vector in sorted_data])

print("国名数:", len(country_names), flush=True)
print("国名リストとベクトル分け完了", flush=True)

# 描画・t-SNE前に巨大な学習済みモデルを解放する
del model
gc.collect()
print("モデル解放完了", flush=True)

# t-SNEを使って、300次元の国名ベクトルを2次元に圧縮する
# n_components=2 は2次元にするという意味
# random_state=0 は実行ごとの結果のばらつきを抑えるため
print("t-SNE開始", flush=True)

tsne = TSNE(
    n_components=2,
    random_state=0,
    init="random",
    learning_rate="auto",
)

country_vectors_2d = tsne.fit_transform(country_vectors)

print("t-SNE完了", flush=True)

# 図を作成する
fig, ax = plt.subplots(figsize=(14, 10))

print("散布図描画開始", flush=True)

# 各国を2次元平面上にプロットする
for country, vector_2d in zip(country_names, country_vectors_2d):
    # t-SNE後の2次元座標を取り出す
    x = vector_2d[0]
    y = vector_2d[1]

    # 点を描く
    ax.scatter(x, y)

    # 点の近くに国名を表示する
    ax.text(x, y, country, fontsize=8)

# 保存先を明示する
output_file = Path(__file__).with_name("answer_59_tsne.png")

# グラフのタイトルと軸ラベルを設定する
ax.set_title("t-SNE Visualization of Country Vectors")
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")

# レイアウトを調整する
fig.tight_layout()

print("画像を保存します...", flush=True)

# 画像として保存する
fig.savefig(output_file, dpi=200)

# メモリ解放のため、図を閉じる
plt.close(fig)

print("保存処理が終わりました。", flush=True)
print(f"結果は {output_file} に保存されています。", flush=True)

# 出力結果は"answer_59_tsne.png"に保存されている
