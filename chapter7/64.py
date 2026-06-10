import csv
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 64.py が置かれているフォルダを取得する
BASE_DIR = Path(__file__).parent

# 64.py と同じ場所にある SST-2 フォルダを指定する
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

    # 各単語の出現回数を数え、普通の辞書に変換して返す
    return dict(Counter(tokens))


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

            # 文をBoW特徴ベクトルに変換する
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
# 特徴ベクトルを数値行列に変換する
# ==============================

# 辞書形式の特徴ベクトルを数値行列に変換する道具を用意する
vectorizer = DictVectorizer()

# trainデータを見て単語と列番号の対応を覚えつつ、数値行列に変換する
X_train = vectorizer.fit_transform(X_train_dict)


# ==============================
# ロジスティック回帰モデルを学習する
# ==============================

model = LogisticRegression(max_iter=1000)

# 学習データを使ってモデルを学習する
model.fit(X_train, y_train)


# ==============================
# 検証データの先頭事例を取り出す
# ==============================

# dev.tsv の先頭の事例を取り出す
first_dev_example = dev_data[0]

# 先頭事例の文
first_text = first_dev_example["text"]

# 先頭事例の正解ラベル
gold_label = first_dev_example["label"]

# 先頭事例の特徴ベクトル
first_feature = first_dev_example["feature"]


# ==============================
# 検証データの先頭事例を数値行列に変換する
# ==============================

# 1件だけでも、DictVectorizerには「辞書のリスト」として渡す必要がある
# そのため first_feature ではなく [first_feature] と書く
#
# ここでは fit_transform ではなく transform を使う
# 理由:
#   trainデータで覚えた単語と列番号の対応をそのまま使う必要があるため
X_dev_first = vectorizer.transform([first_feature])


# ==============================
# 予測ラベルと予測確率を求める
# ==============================

# predict は、最終的な予測ラベルを返す
# 例: [1]
pred_label = model.predict(X_dev_first)[0]

# predict_proba は、各ラベルに属する確率を返す
# 例: [[0.12, 0.88]]
# 1件だけ予測しているので、[0] で先頭の結果を取り出す
pred_proba = model.predict_proba(X_dev_first)[0]


# ==============================
# 結果を表示する
# ==============================

print("===== 検証データの先頭事例 =====")
print(f"文: {first_text}")
print(f"正解ラベル: {gold_label}")
print(f"予測ラベル: {pred_label}")

print()

print("===== 条件付き確率 =====")

for label, probability in zip(model.classes_, pred_proba):
    print(f"P(y = {label} | x) = {probability:.4f}")

print()

if pred_label == gold_label:
    print("予測は正解です。")
else:
    print("予測は不正解です。")

# 出力
# ===== 検証データの先頭事例 =====
# 文: it 's a charming and often affecting journey .
# 正解ラベル: 1
# 予測ラベル: 1

# ===== 条件付き確率 =====
# P(y = 0 | x) = 0.0042
# P(y = 1 | x) = 0.9958

# 予測は正解です。
