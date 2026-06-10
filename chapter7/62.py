import csv
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 62.py が置かれているフォルダを取得する
BASE_DIR = Path(__file__).parent

# 62.py と同じ場所にある SST-2 フォルダを指定する
SST2_DIR = BASE_DIR / "SST-2"

# 学習データ train.tsv の場所
train_path = SST2_DIR / "train.tsv"


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

    # 文をスペースで分割してトークン列にする
    tokens = text.split()

    # 各トークンの出現回数を数え、普通の辞書に変換して返す
    return dict(Counter(tokens))


# ==============================
# TSVファイルを読み込む関数
# ==============================


def load_dataset(tsv_path):
    """
    train.tsv を読み込み、
    各事例を {'text', 'label', 'feature'} を持つ辞書に変換する。
    """

    # 変換後の事例を入れるリスト
    dataset = []

    # TSVファイルを開く
    with open(tsv_path, "r", encoding="utf-8") as f:

        # TSVはタブ区切りなので delimiter="\t" を指定する
        reader = csv.DictReader(f, delimiter="\t")

        # 1行ずつ処理する
        for row in reader:

            # 英文を取り出す
            text = row["sentence"]

            # 正解ラベルを取り出す
            # ここでは後で扱いやすいように int に変換する
            # "0" -> 0, "1" -> 1
            label = int(row["label"])

            # 英文をBoW特徴ベクトルに変換する
            feature = create_feature(text)

            # 1つの事例を辞書としてまとめる
            example = {"text": text, "label": label, "feature": feature}

            # リストに追加する
            dataset.append(example)

    return dataset


# ==============================
# 学習データを読み込む
# ==============================

train_data = load_dataset(train_path)


# ==============================
# X_train と y_train を作る
# ==============================

# X_train_dict は、特徴ベクトルの辞書だけを集めたリスト
# 例:
# [
#   {'too': 2, 'loud': 1, ',': 1, 'goofy': 1},
#   {'a': 1, 'touching': 1, ',': 1, 'thoughtful': 1, 'film': 1},
#   ...
# ]
X_train_dict = [example["feature"] for example in train_data]

# y_train は、正解ラベルだけを集めたリスト
# 例: [0, 1, 0, 1, ...]
y_train = [example["label"] for example in train_data]


# ==============================
# 辞書形式の特徴ベクトルを、数値行列に変換する
# ==============================

# DictVectorizer は、辞書のリストを scikit-learn が扱える数値行列に変換する道具
vectorizer = DictVectorizer()

# fit_transform は、
# 1. 学習データに出てくる全特徴量、つまり全単語の一覧を覚える
# 2. 各文の特徴辞書を数値行列に変換する
# という2つを同時に行う
X_train = vectorizer.fit_transform(X_train_dict)


# ==============================
# ロジスティック回帰モデルを学習する
# ==============================

# max_iter は、学習の最大繰り返し回数
# デフォルトだと収束しきらず警告が出ることがあるので、少し大きめにしている
model = LogisticRegression(max_iter=1000)

# fit に X_train と y_train を渡して学習する
model.fit(X_train, y_train)


# ==============================
# 学習結果を確認する
# ==============================

print("ロジスティック回帰モデルの学習が完了しました。")

# 学習データの事例数
print(f"学習データの事例数: {len(train_data)}")

# 特徴量の数
# ここでは、学習データに出てきた単語の種類数に近い
print(f"特徴量の数: {len(vectorizer.get_feature_names_out())}")

# モデルが分類できるラベル
print(f"分類ラベル: {model.classes_}")

# 出力
# ロジスティック回帰モデルの学習が完了しました。
# 学習データの事例数: 67349
# 特徴量の数: 14816
# 分類ラベル: [0 1]
