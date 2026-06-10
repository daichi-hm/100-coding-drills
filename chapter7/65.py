import csv
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 65.py が置かれているフォルダを取得する
BASE_DIR = Path(__file__).parent

# 65.py と同じ場所にある SST-2 フォルダを指定する
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
        "the worst movie I 've ever seen"
        -> {'the': 1, 'worst': 1, 'movie': 1, 'I': 1, "'ve": 1, 'ever': 1, 'seen': 1}
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

            # ラベルは "0" や "1" の文字列なので、整数に変換する
            label = int(row["label"])

            # 文をBoW特徴ベクトルに変換する
            feature = create_feature(text)

            example = {"text": text, "label": label, "feature": feature}

            dataset.append(example)

    return dataset


# ==============================
# 入力テキストのポジネガを予測する関数
# ==============================


def predict_sentiment(text, vectorizer, model):
    """
    与えられた英文 text のポジネガを予測する関数。

    引数:
        text:
            予測したい英文

        vectorizer:
            学習データで fit 済みの DictVectorizer

        model:
            学習済みのロジスティック回帰モデル

    戻り値:
        pred_label:
            予測ラベル
            0 = ネガティブ
            1 = ポジティブ

        pred_proba:
            各ラベルの条件付き確率
    """

    # 入力文を BoW 特徴ベクトルに変換する
    feature = create_feature(text)

    # DictVectorizer には「辞書のリスト」を渡す必要がある
    # 今回は1文だけなので [feature] のようにリストで包む
    #
    # 注意:
    # ここでは fit_transform ではなく transform を使う
    # 学習時に覚えた単語と列番号の対応をそのまま使うため
    X = vectorizer.transform([feature])

    # 予測ラベルを求める
    pred_label = model.predict(X)[0]

    # 各ラベルに分類される条件付き確率を求める
    pred_proba = model.predict_proba(X)[0]

    return pred_label, pred_proba


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

# 辞書形式の特徴ベクトルを数値行列に変換する道具を用意する
vectorizer = DictVectorizer()

# trainデータを見て、単語と列番号の対応を覚えながら数値行列に変換する
X_train = vectorizer.fit_transform(X_train_dict)


# ==============================
# ロジスティック回帰モデルを学習する
# ==============================

model = LogisticRegression(max_iter=1000)

# 学習データを使ってモデルを学習する
model.fit(X_train, y_train)


# ==============================
# 自分で与えるテキスト
# ==============================

# 問題文の例
input_text = "the worst movie I 've ever seen"


# ==============================
# 入力テキストのポジネガを予測する
# ==============================

pred_label, pred_proba = predict_sentiment(input_text, vectorizer, model)


# ==============================
# 結果を表示する
# ==============================

print("===== 入力テキスト =====")
print(input_text)

print()

print("===== 予測結果 =====")
print(f"予測ラベル: {pred_label}")

# ラベルを人間に分かりやすい名前に変換する
if pred_label == 0:
    print("判定: ネガティブ")
else:
    print("判定: ポジティブ")

print()

print("===== 条件付き確率 =====")

# model.classes_ と pred_proba は同じ順番で対応している
# 例:
#   model.classes_ = [0, 1]
#   pred_proba     = [0.98, 0.02]
#
# これは、
#   P(y=0 | x) = 0.98
#   P(y=1 | x) = 0.02
# という意味
for label, probability in zip(model.classes_, pred_proba):
    print(f"P(y = {label} | x) = {probability:.4f}")

# 出力
# ===== 入力テキスト =====
# the worst movie I 've ever seen

# ===== 予測結果 =====
# 予測ラベル: 0
# 判定: ネガティブ

# ===== 条件付き確率 =====
# P(y = 0 | x) = 0.9788
# P(y = 1 | x) = 0.0212
