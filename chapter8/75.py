from pathlib import Path
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =========================
# パス設定
# =========================

# このファイルが置かれているディレクトリ
# 例: nlp100/chapter8
BASE_DIR = Path(__file__).resolve().parent

# プロジェクト全体のルートディレクトリ
# 例: nlp100
PROJECT_ROOT = BASE_DIR.parent

# 70番で作成した単語埋め込みデータ
# 例: nlp100/chapter8/data/embeddings.pt
EMBEDDING_FILE = BASE_DIR / "data" / "embeddings.pt"

# 71番で使ったSST-2の訓練データ
# 例: nlp100/chapter7/SST-2/train.tsv
TRAIN_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "train.tsv"


# =========================
# DataLoader設定
# =========================

# 75番では動作確認しやすいように4件ずつ取り出す
BATCH_SIZE = 4


def tokenize(text):
    """
    文をトークン列に変換する関数。

    71番・73番・74番と同じように、
    小文字化してスペースで分割する。
    """

    text = text.lower()
    tokens = text.split()

    return tokens


def text_to_token_ids(text, token_to_id):
    """
    文をトークンID列に変換する関数。

    token_to_id に存在しない単語は無視する。
    """

    tokens = tokenize(text)

    token_ids = []

    for token in tokens:
        if token in token_to_id:
            token_ids.append(token_to_id[token])

    return token_ids


def load_sst_file(file_path, token_to_id):
    """
    SST-2のtsvファイルを読み込み、
    各文をトークンID列に変換する関数。
    """

    examples = []

    total_count = 0
    removed_count = 0

    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            total_count += 1

            text = row["sentence"]
            label = int(row["label"])

            input_ids = text_to_token_ids(text, token_to_id)

            # token_to_idに含まれる単語が1つもない文は削除する
            if len(input_ids) == 0:
                removed_count += 1
                continue

            examples.append(
                {
                    "text": text,
                    "label": label,
                    "input_ids": input_ids,
                }
            )

    print(f"{file_path.name}: 元データ数 = {total_count}")
    print(f"{file_path.name}: 削除したデータ数 = {removed_count}")
    print(f"{file_path.name}: 使用するデータ数 = {len(examples)}")

    return examples


class SSTDataset(Dataset):
    """
    SST-2をPyTorchで扱うための自作Datasetクラス。
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)

        # 73番・74番と同じく、CrossEntropyLossで扱いやすいようにlong型にする
        label = torch.tensor(example["label"], dtype=torch.long)

        return {
            "text": example["text"],
            "input_ids": input_ids,
            "label": label,
        }


def collate_batch(batch):
    """
    複数の事例を1つのミニバッチにまとめる関数。

    75番でやること:
    1. input_ids の長さが長い順に事例を並び替える
    2. 一番長いinput_idsの長さに合わせて、短い文を0でpaddingする
    3. input_ids, label, lengths をテンソルとして返す
    """

    # =========================
    # 1. 長さが長い順に並び替える
    # =========================
    #
    # batch は、Datasetから取り出された複数事例のリスト。
    #
    # 例:
    # [
    #   {"text": ..., "input_ids": tensor([...]), "label": tensor(0)},
    #   {"text": ..., "input_ids": tensor([...]), "label": tensor(1)},
    #   ...
    # ]
    #
    # len(example["input_ids"]) が文のトークン数。
    # これが大きい順に並べる。
    sorted_batch = sorted(
        batch,
        key=lambda example: len(example["input_ids"]),
        reverse=True,
    )

    texts = []
    input_ids_list = []
    labels = []
    lengths = []

    # =========================
    # 2. 並び替え後の順番で、それぞれの情報を集める
    # =========================

    for example in sorted_batch:
        texts.append(example["text"])
        input_ids_list.append(example["input_ids"])
        labels.append(example["label"])

        # padding前の本来の長さ
        lengths.append(len(example["input_ids"]))

    # =========================
    # 3. input_idsをpaddingする
    # =========================
    #
    # input_ids_list は、長さがバラバラのテンソルのリスト。
    #
    # 例:
    # [
    #   tensor([4, 5053, 45, 3305]),
    #   tensor([5785, 66]),
    #   tensor([3475, 87, 15888]),
    # ]
    #
    # pad_sequenceを使うと、一番長い系列に長さをそろえ、
    # 足りない部分を padding_value=0 で埋める。
    padded_input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0,
    )

    # =========================
    # 4. labelsを1つのテンソルにまとめる
    # =========================
    #
    # labels は最初、
    #   [tensor(1), tensor(0), tensor(0), tensor(1)]
    # のようなリスト。
    #
    # torch.stackすると、
    #   tensor([1, 0, 0, 1])
    # になる。
    labels = torch.stack(labels)

    # =========================
    # 5. lengthsをテンソルにする
    # =========================
    #
    # lengths は最初、
    #   [11, 7, 7, 6]
    # のようなPythonリスト。
    #
    # これをPyTorchで扱えるようにテンソルにする。
    lengths = torch.tensor(lengths, dtype=torch.long)

    return {
        "text": texts,
        "input_ids": padded_input_ids,
        "label": labels,
        "lengths": lengths,
    }


def main():
    print("padding付きcollate関数を実装します。")

    # =========================
    # 70番の単語埋め込みデータを読み込む
    # =========================

    data = torch.load(EMBEDDING_FILE, map_location="cpu")

    token_to_id = data["token_to_id"]

    print(f"語彙数: {len(token_to_id)}")

    # =========================
    # 訓練データを読み込む
    # =========================

    train_examples = load_sst_file(TRAIN_FILE, token_to_id)

    train_dataset = SSTDataset(train_examples)

    print(f"train_dataset のデータ数: {len(train_dataset)}")

    # =========================
    # 先頭4件を手動でcollate_batchに渡して確認する
    # =========================
    #
    # 公式問題の例に近い確認方法。
    # DataLoaderを使わずに、まずcollate_batch単体の動きを見る。
    first_four_examples = [
        train_dataset[0],
        train_dataset[1],
        train_dataset[2],
        train_dataset[3],
    ]

    batch = collate_batch(first_four_examples)

    print("\n先頭4件をcollate_batchに渡した結果:")
    print("text:")
    for text in batch["text"]:
        print(f"  {text}")

    print(f"\ninput_ids:")
    print(batch["input_ids"])

    print(f"\nlabel:")
    print(batch["label"])

    print(f"\nlengths:")
    print(batch["lengths"])

    print(f"\ninput_ids の形: {batch['input_ids'].shape}")
    print(f"label の形: {batch['label'].shape}")
    print(f"lengths の形: {batch['lengths'].shape}")

    # =========================
    # DataLoaderでも同じcollate_batchを使えることを確認する
    # =========================
    #
    # shuffle=Falseにしておくと、先頭から順番に取り出されるので確認しやすい。
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    loader_batch = next(iter(train_loader))

    print("\nDataLoaderから取り出した最初の1バッチ:")
    print(f"input_ids の形: {loader_batch['input_ids'].shape}")
    print(f"label の形: {loader_batch['label'].shape}")
    print(f"lengths: {loader_batch['lengths']}")


if __name__ == "__main__":
    main()

# 出力
# padding付きcollate関数を実装します。
# 語彙数: 300001
# train.tsv: 元データ数 = 67349
# train.tsv: 削除したデータ数 = 953
# train.tsv: 使用するデータ数 = 66396
# train_dataset のデータ数: 66396

# 先頭4件をcollate_batchに渡した結果:
# text:
#   that loves its characters and communicates something rather beautiful about human nature 
#   hide new secretions from the parental units 
#   remains utterly satisfied to remain the same throughout 
#   contains no wit , only labored gags 

# input_ids:
# tensor([[     4,   5053,     45,   3305,  31647,    348,    904,   2815,     47,
#            1276,   1964],
#         [  5785,     66, 113845,     18,     12,  15095,   1594,      0,      0,
#               0,      0],
#         [   987,  14528,   4941,    873,     12,    208,    898,      0,      0,
#               0,      0],
#         [  3475,     87,  15888,     90,  27695,  42637,      0,      0,      0,
#               0,      0]])

# label:
# tensor([1, 0, 0, 0])

# lengths:
# tensor([11,  7,  7,  6])

# input_ids の形: torch.Size([4, 11])
# label の形: torch.Size([4])
# lengths の形: torch.Size([4])

# DataLoaderから取り出した最初の1バッチ:
# input_ids の形: torch.Size([4, 11])
# label の形: torch.Size([4])
# lengths: tensor([11,  7,  7,  6])
