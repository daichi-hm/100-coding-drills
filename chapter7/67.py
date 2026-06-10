import csv
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 67.py が置かれているフォルダを取得する
BASE_DIR = Path(__file__).parent

# 67.py と同じ場所にある SST-2 フォルダを指定する
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

            # 文をBoW特徴ベクトルに変換する
            feature = create_feature(text)

            example = {"text": text, "label": label, "feature": feature}

            dataset.append(example)

    return dataset


# ==============================
# 評価指標を計算して表示する関数
# ==============================


def evaluate(y_true, y_pred, dataset_name):
    """
    正解ラベルと予測ラベルから、
    正解率・適合率・再現率・F1スコアを計算して表示する関数。

    引数:
        y_true:
            正解ラベルのリスト

        y_pred:
            モデルが予測したラベルのリスト

        dataset_name:
            表示用のデータセット名
            例: "学習データ", "検証データ"
    """

    # 正解率
    accuracy = accuracy_score(y_true, y_pred)

    # 適合率
    # pos_label=1 は、「ポジティブクラスを 1 として計算する」という意味
    precision = precision_score(y_true, y_pred, pos_label=1)

    # 再現率
    recall = recall_score(y_true, y_pred, pos_label=1)

    # F1スコア
    f1 = f1_score(y_true, y_pred, pos_label=1)

    print(f"===== {dataset_name} の評価 =====")
    print(f"正解率 accuracy : {accuracy:.4f}")
    print(f"適合率 precision: {precision:.4f}")
    print(f"再現率 recall   : {recall:.4f}")
    print(f"F1スコア f1     : {f1:.4f}")
    print()


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

# 学習データでは fit_transform を使う
# 学習データに出てくる単語と列番号の対応を覚えつつ、数値行列に変換する
X_train = vectorizer.fit_transform(X_train_dict)

# 検証データでは transform だけを使う
# 学習データで覚えた単語と列番号の対応をそのまま使う
X_dev = vectorizer.transform(X_dev_dict)


# ==============================
# ロジスティック回帰モデルを学習する
# ==============================

model = LogisticRegression(max_iter=1000)

# 学習データを使ってモデルを学習する
model.fit(X_train, y_train)


# ==============================
# 学習データと検証データを予測する
# ==============================

# 学習データに対する予測
y_train_pred = model.predict(X_train)

# 検証データに対する予測
y_dev_pred = model.predict(X_dev)


# ==============================
# 評価指標を計算する
# ==============================

evaluate(y_train, y_train_pred, "学習データ")
evaluate(y_dev, y_dev_pred, "検証データ")

# 出力
# ===== 学習データ の評価 =====
# 正解率 accuracy : 0.9428
# 適合率 precision: 0.9434
# 再現率 recall   : 0.9547
# F1スコア f1     : 0.9490

# ===== 検証データ の評価 =====
# 正解率 accuracy : 0.8108
# 適合率 precision: 0.7987
# 再現率 recall   : 0.8401
# F1スコア f1     : 0.8189
