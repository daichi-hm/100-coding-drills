import csv
from pathlib import Path
from collections import Counter

import matplotlib

# WSLやターミナル環境では、plt.show()でグラフが表示されないことがある
# そのため、画面表示ではなく画像ファイルとして保存する設定にする
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 69.py が置かれているフォルダを取得する
BASE_DIR = Path(__file__).parent

# 69.py と同じ場所にある SST-2 フォルダを指定する
SST2_DIR = BASE_DIR / "SST-2"

# 学習データ train.tsv の場所
train_path = SST2_DIR / "train.tsv"

# 検証データ dev.tsv の場所
dev_path = SST2_DIR / "dev.tsv"


# ==============================
# 特徴ベクトルを作る関数
# ==============================


def create_feature(text):
    """
    1つの英文を BoW の特徴ベクトルに変換する関数。

    例:
        "too loud , too goofy"
        -> {'too': 2, 'loud': 1, ',': 1, 'goofy': 1}
    """

    # 文をスペースで分割する
    tokens = text.split()

    # 各トークンの出現回数を数える
    feature = Counter(tokens)

    # Counterを普通の辞書に変換して返す
    return dict(feature)


# ==============================
# TSVファイルを読み込む関数
# ==============================


def load_dataset(tsv_path):
    """
    train.tsv や dev.tsv を読み込み、
    各事例を以下の形の辞書に変換する関数。

    {
        "text": 文,
        "label": ラベル,
        "feature": 特徴ベクトル
    }
    """

    dataset = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]

            # ラベルは "0" や "1" の文字列なので、整数に変換する
            label = int(row["label"])

            # 文を BoW 特徴ベクトルに変換する
            feature = create_feature(text)

            example = {"text": text, "label": label, "feature": feature}

            dataset.append(example)

    return dataset


# ==============================
# train.tsv と dev.tsv を読み込む
# ==============================

train_data = load_dataset(train_path)
dev_data = load_dataset(dev_path)


# ==============================
# 学習データから X_train と y_train を作る
# ==============================

# 学習データの特徴ベクトルだけを取り出す
X_train_dict = [example["feature"] for example in train_data]

# 学習データの正解ラベルだけを取り出す
y_train = [example["label"] for example in train_data]


# ==============================
# 検証データから X_dev と y_dev を作る
# ==============================

# 検証データの特徴ベクトルだけを取り出す
X_dev_dict = [example["feature"] for example in dev_data]

# 検証データの正解ラベルだけを取り出す
y_dev = [example["label"] for example in dev_data]


# ==============================
# 特徴ベクトルを数値行列に変換する
# ==============================

# 辞書形式の特徴ベクトルを数値行列に変換する道具
vectorizer = DictVectorizer()

# trainデータでは fit_transform を使う
# fit:
#   学習データに出てくる単語と列番号の対応を覚える
# transform:
#   その対応に従って数値行列に変換する
X_train = vectorizer.fit_transform(X_train_dict)

# devデータでは transform だけを使う
# trainデータで覚えた単語と列番号の対応をそのまま使う
X_dev = vectorizer.transform(X_dev_dict)


# ==============================
# 正則化パラメータ C を変えながら実験する
# ==============================

# scikit-learn の LogisticRegression では、
# C が小さいほど正則化が強く、
# C が大きいほど正則化が弱い
C_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# 各 C に対する検証データ上の正解率を保存するリスト
dev_accuracies = []


# C を1つずつ変えてモデルを学習・評価する
for C in C_values:
    print(f"C = {C} で学習中...")

    # ロジスティック回帰モデルを作る
    # C によって正則化の強さを変える
    model = LogisticRegression(C=C, max_iter=1000, solver="liblinear")

    # 学習データでモデルを学習する
    model.fit(X_train, y_train)

    # 検証データに対する予測を行う
    y_dev_pred = model.predict(X_dev)

    # 検証データ上の正解率を計算する
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)

    # 結果をリストに保存する
    dev_accuracies.append(dev_accuracy)

    # 結果をターミナルに表示する
    print(f"  検証データの正解率: {dev_accuracy:.4f}")


# ==============================
# 一番よかった C を確認する
# ==============================

# 検証データの正解率が最大になる位置を調べる
best_index = dev_accuracies.index(max(dev_accuracies))

# その位置に対応する C と正解率を取り出す
best_C = C_values[best_index]
best_accuracy = dev_accuracies[best_index]

print()
print("===== 最良の結果 =====")
print(f"最も検証データの正解率が高かった C: {best_C}")
print(f"検証データの正解率: {best_accuracy:.4f}")


# ==============================
# グラフを作成して画像として保存する
# ==============================

# グラフのサイズを指定する
plt.figure(figsize=(8, 6))

# 横軸 C、縦軸 検証データの正解率でグラフを描く
# marker="o" は、各点に丸印を付ける設定
plt.plot(C_values, dev_accuracies, marker="o")

# C は広い範囲で変化するため、横軸を対数スケールにする
plt.xscale("log")

# グラフタイトルと軸ラベル
plt.title("Effect of regularization parameter C")
plt.xlabel("C (inverse regularization strength)")
plt.ylabel("Dev accuracy")

# グリッドを表示する
plt.grid(True)

# y軸の範囲を見やすくする
plt.ylim(0.5, 1.0)

# 画像として保存する
output_file = "69_regularization_dev_accuracy.png"
plt.savefig(output_file, bbox_inches="tight")

print()
print(f"{output_file} を保存しました。")

# 出力画像は"69_regularization_dev_accuracy.png"に入っている
# 出力
# C = 0.001 で学習中...
#   検証データの正解率: 0.6846
# C = 0.003 で学習中...
#   検証データの正解率: 0.7110
# C = 0.01 で学習中...
#   検証データの正解率: 0.7557
# C = 0.03 で学習中...
#   検証データの正解率: 0.7947
# C = 0.1 で学習中...
#   検証データの正解率: 0.8005
# C = 0.3 で学習中...
#   検証データの正解率: 0.8062
# C = 1 で学習中...
#   検証データの正解率: 0.8108
# C = 3 で学習中...
#   検証データの正解率: 0.8211
# C = 10 で学習中...
#   検証データの正解率: 0.8245
# C = 30 で学習中...
#   検証データの正解率: 0.8096
# C = 100 で学習中...
#   検証データの正解率: 0.7947

# ===== 最良の結果 =====
# 最も検証データの正解率が高かった C: 10
# 検証データの正解率: 0.8245

# 69_regularization_dev_accuracy.png を保存しました。
