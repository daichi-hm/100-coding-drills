import csv
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 63.py が置かれているフォルダを取得する
BASE_DIR = Path(__file__).parent

# 63.py と同じ場所にある SST-2 フォルダを指定する
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

            # 今回は比較しやすいように、ラベルを整数に変換する
            # "0" -> 0
            # "1" -> 1
            label = int(row["label"])

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

# 学習データに出てくる単語を覚えながら、数値行列に変換する
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
# 検証データの先頭事例を予測する
# ==============================

# 注意:
# vectorizer.transform には「辞書のリスト」を渡す必要がある
# first_feature は1件分の辞書なので、[first_feature] のようにリストで包む
X_dev_first = vectorizer.transform([first_feature])

# 学習済みモデルを使って予測する
# predict の戻り値は配列なので、[0] で先頭の予測結果を取り出す
pred_label = model.predict(X_dev_first)[0]


# ==============================
# 結果を表示する
# ==============================

print("===== 検証データの先頭事例 =====")
print(f"文: {first_text}")
print(f"正解ラベル: {gold_label}")
print(f"予測ラベル: {pred_label}")

print()

# 予測ラベルと正解ラベルが一致しているか確認する
if pred_label == gold_label:
    print("予測は正解です。")
else:
    print("予測は不正解です。")

# 出力
# ===== 検証データの先頭事例 =====
# 文: it 's a charming and often affecting journey .
# 正解ラベル: 1
# 予測ラベル: 1

# 予測は正解です。
