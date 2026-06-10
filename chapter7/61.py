import csv
from pathlib import Path
from collections import Counter

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 61.py が置かれているフォルダを取得する
# 例: chapter7/61.py なら、BASE_DIR は chapter7/ になる
BASE_DIR = Path(__file__).parent

# 61.py と同じ場所にある SST-2 フォルダを指定する
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

    引数:
        text:
            英文の文字列
            例: "too loud , too goofy"

    戻り値:
        feature:
            単語の出現回数を表す辞書
            例: {'too': 2, 'loud': 1, ',': 1, 'goofy': 1}
    """

    # text.split() は、文字列をスペースで分割する
    # 例: "too loud , too goofy"
    #     -> ['too', 'loud', ',', 'too', 'goofy']
    tokens = text.split()

    # Counter は、同じ値が何回出てきたかを数える道具
    # 例: ['too', 'loud', ',', 'too', 'goofy']
    #     -> Counter({'too': 2, 'loud': 1, ',': 1, 'goofy': 1})
    feature = Counter(tokens)

    # Counter は辞書に近い形だが、見た目を問題文に近づけるため dict に変換する
    return dict(feature)


# ==============================
# TSVファイルを読み込んで、事例のリストを作る関数
# ==============================


def load_dataset(tsv_path):
    """
    train.tsv や dev.tsv を読み込み、
    各事例を以下の形の辞書に変換する関数。

    {
        'text': 文,
        'label': ラベル,
        'feature': 特徴ベクトル
    }

    引数:
        tsv_path:
            train.tsv や dev.tsv のファイルパス

    戻り値:
        dataset:
            辞書オブジェクトのリスト
    """

    # 変換した事例を入れるための空リスト
    dataset = []

    # TSVファイルを開く
    with open(tsv_path, "r", encoding="utf-8") as f:

        # TSVはタブ区切りなので delimiter="\t" を指定する
        # DictReaderを使うと、row["sentence"] や row["label"] で値を取り出せる
        reader = csv.DictReader(f, delimiter="\t")

        # 1行ずつ読み込む
        for row in reader:

            # sentence列から英文を取り出す
            text = row["sentence"]

            # label列からラベルを取り出す
            # ラベルは "0" または "1" の文字列
            label = row["label"]

            # 英文をBoW特徴ベクトルに変換する
            feature = create_feature(text)

            # 1つの事例を辞書としてまとめる
            example = {"text": text, "label": label, "feature": feature}

            # 作った事例をリストに追加する
            dataset.append(example)

    # 全事例をまとめたリストを返す
    return dataset


# ==============================
# train.tsv と dev.tsv を読み込む
# ==============================

train_data = load_dataset(train_path)
dev_data = load_dataset(dev_path)


# ==============================
# 結果を確認する
# ==============================

# 学習データと検証データの件数を表示する
print(f"学習データの事例数: {len(train_data)}")
print(f"検証データの事例数: {len(dev_data)}")

print()

# 学習データの最初の事例を表示する
# 問題文で「学習データの最初の事例について、目視で確認せよ」とあるため
print("===== 学習データの最初の事例 =====")
print(train_data[0])

# 出力
# 学習データの事例数: 67349
# 検証データの事例数: 872

# ===== 学習データの最初の事例 =====
# {'text': 'hide new secretions from the parental units ', 'label': '0', 'feature': {'hide': 1, 'new': 1, 'secretions': 1, 'from': 1, 'the': 1, 'parental': 1, 'units': 1}}
