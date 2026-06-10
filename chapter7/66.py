import csv
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 66.py が置かれているフォルダを取得する
BASE_DIR = Path(__file__).parent

# 66.py と同じ場所にある SST-2 フォルダを指定する
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

            # "0" や "1" の文字列を整数に変換する
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
# ロジスティック回帰モデルを学習する
# ==============================

model = LogisticRegression(max_iter=1000)

# 学習データでモデルを学習する
model.fit(X_train, y_train)


# ==============================
# 検証データ全体を予測する
# ==============================

# dev.tsv の全事例に対して予測ラベルを求める
y_pred = model.predict(X_dev)


# ==============================
# 混同行列を作る
# ==============================

# labels=[0, 1] と指定することで、
# 行・列の順番を 0, 1 に固定する
cm = confusion_matrix(y_dev, y_pred, labels=[0, 1])


# ==============================
# 結果を表示する
# ==============================
print("===== 混同行列 =====")
print("行: 正解ラベル")
print("列: 予測ラベル")
print()
print("          予測0    予測1")
print(f"正解0     {cm[0][0]:5d}   {cm[0][1]:5d}")
print(f"正解1     {cm[1][0]:5d}   {cm[1][1]:5d}")

# 出力
# ===== 混同行列 =====
# 行: 正解ラベル
# 列: 予測ラベル

#           予測0    予測1
# 正解0       334      94
# 正解1        71     373
