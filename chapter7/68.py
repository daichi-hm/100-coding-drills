import csv
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 68.py が置かれているフォルダを取得する
BASE_DIR = Path(__file__).parent

# 68.py と同じ場所にある SST-2 フォルダを指定する
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
    train.tsv を読み込み、
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
# 学習データを読み込む
# ==============================

train_data = load_dataset(train_path)


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

# 辞書形式の特徴ベクトルを数値行列に変換する道具
vectorizer = DictVectorizer()

# 学習データに出てくる単語と列番号の対応を覚えながら、数値行列に変換する
X_train = vectorizer.fit_transform(X_train_dict)


# ==============================
# ロジスティック回帰モデルを学習する
# ==============================

model = LogisticRegression(max_iter=1000)

# 学習データを使ってモデルを学習する
model.fit(X_train, y_train)


# ==============================
# 特徴量名と重みを取り出す
# ==============================

# DictVectorizer が覚えた特徴量名、つまり単語一覧を取り出す
# 例: ['!' ',' 'bad' 'excellent' 'movie' ...]
feature_names = vectorizer.get_feature_names_out()

# ロジスティック回帰モデルが学習した重みを取り出す
# 2値分類の場合、model.coef_ は [[重み1, 重み2, ...]] のような2次元配列
# そのため [0] を付けて、1次元の重み配列にする
weights = model.coef_[0]


# ==============================
# 特徴量名と重みをセットにする
# ==============================

# feature_names と weights は同じ順番で対応している
# 例:
#   feature_names[0] に対応する重みが weights[0]
#   feature_names[1] に対応する重みが weights[1]
#
# zip で単語と重みをペアにする
feature_weights = list(zip(feature_names, weights))


# ==============================
# 重みに基づいて並べ替える
# ==============================

# 重みが大きい順に並べ替える
# x[1] は、(特徴量名, 重み) のうち「重み」を表す
high_weight_features = sorted(feature_weights, key=lambda x: x[1], reverse=True)

# 重みが小さい順に並べ替える
low_weight_features = sorted(feature_weights, key=lambda x: x[1])


# ==============================
# 結果を表示する
# ==============================

print("===== 重みの高い特徴量トップ20 =====")
print("ポジティブ判定に強く効く特徴量")
for feature, weight in high_weight_features[:20]:
    print(f"{feature}\t{weight:.4f}")

print()

print("===== 重みの低い特徴量トップ20 =====")
print("ネガティブ判定に強く効く特徴量")
for feature, weight in low_weight_features[:20]:
    print(f"{feature}\t{weight:.4f}")

# 出力
# ===== 重みの高い特徴量トップ20 =====
# ポジティブ判定に強く効く特徴量
# refreshing      3.4795
# remarkable      3.4742
# powerful        3.2543
# hilarious       3.1922
# beautiful       3.0081
# wonderful       2.9953
# prose   2.9324
# terrific        2.9235
# treat   2.8796
# appealing       2.8634
# enjoyable       2.8293
# charmer 2.7811
# vividly 2.7225
# likable 2.7051
# charming        2.6362
# solid   2.6356
# impressive      2.6157
# fascinating     2.5873
# half-bad        2.5869
# intriguing      2.5581

# ===== 重みの低い特徴量トップ20 =====
# ネガティブ判定に強く効く特徴量
# lacking -4.3570
# lacks   -4.0884
# worst   -4.0196
# devoid  -3.7065
# mess    -3.6979
# failure -3.6438
# stupid  -3.3731
# bore    -3.3019
# flat    -3.2291
# depressing      -3.2175
# loses   -3.2165
# waste   -3.1931
# lack    -3.0791
# squanders       -3.0502
# none    -3.0243
# hardly  -3.0192
# lousy   -3.0138
# poor    -3.0134
# pointless       -3.0058
# unfortunately   -2.9683
