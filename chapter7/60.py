import csv
from pathlib import Path
from collections import Counter

# ==============================
# ファイル・フォルダの設定
# ==============================

# この 60.py が置かれているフォルダを取得する
# 例: chapter7/60.py なら、BASE_DIR は chapter7/ になる
BASE_DIR = Path(__file__).parent

# 60.py と同じ場所にある SST-2 フォルダを指定する
# 例: chapter7/SST-2
SST2_DIR = BASE_DIR / "SST-2"

# 学習データ train.tsv の場所
train_path = SST2_DIR / "train.tsv"

# 検証データ dev.tsv の場所
dev_path = SST2_DIR / "dev.tsv"


# ==============================
# ファイルが存在するか確認する
# ==============================

# SST-2 フォルダがなければエラーを出す
if not SST2_DIR.exists():
    raise FileNotFoundError(
        "SST-2 フォルダが見つかりません。\n"
        "60.py と同じ場所に、解凍済みの SST-2 フォルダを置いてください。"
    )

# train.tsv がなければエラーを出す
if not train_path.exists():
    raise FileNotFoundError(
        f"train.tsv が見つかりません。\n" f"探した場所: {train_path}"
    )

# dev.tsv がなければエラーを出す
if not dev_path.exists():
    raise FileNotFoundError(f"dev.tsv が見つかりません。\n" f"探した場所: {dev_path}")


# ==============================
# ラベル数を数える関数
# ==============================


def count_labels(tsv_path):
    """
    TSVファイルを読み込み、label列にある 0 と 1 の数を数える関数。

    引数:
        tsv_path:
            train.tsv や dev.tsv のファイルパス

    戻り値:
        label_counter:
            ラベルごとの出現回数
            例: Counter({'1': 3610, '0': 3310})
    """

    # Counter は、同じ値が何回出てきたかを数えるための道具
    label_counter = Counter()

    # TSVファイルを読み込みモードで開く
    with open(tsv_path, "r", encoding="utf-8") as f:

        # TSVはタブ区切りのファイルなので delimiter="\t" を指定する
        # DictReaderを使うと、列名を使って値を取り出せる
        reader = csv.DictReader(f, delimiter="\t")

        # ファイルを1行ずつ読む
        for row in reader:

            # label列の値を取り出す
            # SST-2では、labelは "0" または "1"
            label = row["label"]

            # そのラベルの出現回数を1増やす
            label_counter[label] += 1

    # 数え終わった結果を返す
    return label_counter


# ==============================
# train.tsv と dev.tsv のラベル数を数える
# ==============================

train_counts = count_labels(train_path)
dev_counts = count_labels(dev_path)


# ==============================
# 結果を表示する
# ==============================

print("===== train.tsv のラベル数 =====")
print(f"ネガティブ 0: {train_counts['0']} 件")
print(f"ポジティブ 1: {train_counts['1']} 件")

print()

print("===== dev.tsv のラベル数 =====")
print(f"ネガティブ 0: {dev_counts['0']} 件")
print(f"ポジティブ 1: {dev_counts['1']} 件")

# 出力
# ===== train.tsv のラベル数 =====
# ネガティブ 0: 29780 件
# ポジティブ 1: 37569 件

# ===== dev.tsv のラベル数 =====
# ネガティブ 0: 428 件
# ポジティブ 1: 444 件
