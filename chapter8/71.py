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

# プロジェクトのルートディレクトリ
# 例: nlp100
PROJECT_ROOT = BASE_DIR.parent

# 70番で作成した単語埋め込みデータ
# 例: nlp100/chapter8/data/embeddings.pt
EMBEDDING_FILE = BASE_DIR / "data" / "embeddings.pt"

# chapter7の中にあるSST-2データ
# 例: nlp100/chapter7/SST-2/train.tsv
TRAIN_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "train.tsv"
DEV_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "dev.tsv"

# DataLoaderで一度に取り出すデータ数
BATCH_SIZE = 4


def tokenize(text):
    """
    文をトークン列に変換する関数。
    例:
        "This movie is good ." -> ["this", "movie", "is", "good", "."]
    """

    text = text.lower()
    tokens = text.split()

    return tokens


def text_to_token_ids(text, token_to_id):
    """
    文をトークンID列に変換する関数。

    例:
        "this movie is good"
    を
        [10, 25, 30, 99]
    のような整数列に変換する。

    token_to_id に存在しない単語は無視する。
    """

    tokens = tokenize(text)

    token_ids = []

    for token in tokens:
        # GoogleNewsの語彙に存在する単語だけIDに変換する
        if token in token_to_id:
            token_ids.append(token_to_id[token])

    return token_ids


def load_sst_file(file_path, token_to_id):
    """
    SST-2のtsvファイルを読み込み、各文をトークンID列に変換する関数。

    戻り値は、以下のような辞書のリスト。

    [
        {
            "text": 元の文,
            "label": ラベル,
            "input_ids": トークンID列
        },
        ...
    ]

    input_ids が空になった事例は削除する。
    """

    examples = []

    total_count = 0
    removed_count = 0

    with file_path.open("r", encoding="utf-8") as f:
        # SST-2はタブ区切りなので delimiter="\t" を指定する
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            total_count += 1

            # SST-2では sentence 列に文、label 列にラベルが入っている
            text = row["sentence"]
            label = int(row["label"])

            # 文をトークンID列に変換する
            input_ids = text_to_token_ids(text, token_to_id)

            # 単語埋め込み語彙に含まれる単語が1つもない事例は削除する
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

    PyTorchのDatasetでは、基本的に以下の3つを実装する。
    1. __init__
    2. __len__
    3. __getitem__
    """

    def __init__(self, examples):
        """
        Dataset作成時に呼ばれる。

        examples:
            load_sst_fileで作成した事例リスト
        """

        self.examples = examples

    def __len__(self):
        """
        データセットの事例数を返す。

        len(train_dataset) のように呼ばれる。
        """

        return len(self.examples)

    def __getitem__(self, idx):
        """
        idx番目の事例を返す。

        train_dataset[0] のように呼ばれる。
        """

        example = self.examples[idx]

        # input_idsは単語ID列なので整数型のテンソルにする
        # nn.Embeddingに入れるIDは torch.long にする必要がある
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)

        # labelも分類ラベルなので整数型のテンソルにする
        label = torch.tensor(example["label"], dtype=torch.long)

        return {
            "text": example["text"],
            "input_ids": input_ids,
            "label": label,
        }


def collate_batch(batch):
    """
    DataLoaderで複数事例を1つのミニバッチにまとめる関数。

    文ごとに input_ids の長さが違うため、
    短い文には <PAD> のIDである0を追加して長さをそろえる。
    """

    texts = []
    input_ids_list = []
    labels = []
    lengths = []

    for example in batch:
        texts.append(example["text"])
        input_ids_list.append(example["input_ids"])
        labels.append(example["label"])
        lengths.append(len(example["input_ids"]))

    # 長さの違うトークンID列を0で埋めて長さをそろえる
    # padding_value=0 は70番で <PAD> 用に予約したID
    padded_input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0,
    )

    # ラベルを1つのテンソルにまとめる
    labels = torch.stack(labels)

    # 各文の元の長さも保存しておく
    # 72番以降で平均を取るときに、<PAD>を除外するために使える
    lengths = torch.tensor(lengths, dtype=torch.long)

    return {
        "text": texts,
        "input_ids": padded_input_ids,
        "label": labels,
        "lengths": lengths,
    }


def main():
    print("SST-2を読み込み、トークンID列に変換します。")
    # =========================
    # 70番で保存したデータを読み込む
    # =========================

    data = torch.load(EMBEDDING_FILE)

    # 71番では、単語からIDへの辞書を使う
    token_to_id = data["token_to_id"]

    print(f"語彙数: {len(token_to_id)}")

    # =========================
    # SST-2のtrain/devを読み込む
    # =========================

    train_examples = load_sst_file(TRAIN_FILE, token_to_id)
    dev_examples = load_sst_file(DEV_FILE, token_to_id)

    # =========================
    # Datasetを作成する
    # =========================

    train_dataset = SSTDataset(train_examples)
    dev_dataset = SSTDataset(dev_examples)

    print(f"train_dataset のデータ数: {len(train_dataset)}")
    print(f"dev_dataset のデータ数: {len(dev_dataset)}")

    # Datasetから1件取り出せるか確認
    print("\ntrain_dataset の先頭1件:")
    print(train_dataset[0])

    # =========================
    # DataLoaderを作成する
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # =========================
    # DataLoaderから1バッチ取り出して確認する
    # =========================

    train_batch = next(iter(train_loader))

    print("\ntrain_loader から取り出した1バッチ:")
    print(f"text: {train_batch['text']}")
    print(f"input_ids の形: {train_batch['input_ids'].shape}")
    print(f"label: {train_batch['label']}")
    print(f"lengths: {train_batch['lengths']}")

    dev_batch = next(iter(dev_loader))

    print("\ndev_loader から取り出した1バッチ:")
    print(f"text: {dev_batch['text']}")
    print(f"input_ids の形: {dev_batch['input_ids'].shape}")
    print(f"label: {dev_batch['label']}")
    print(f"lengths: {dev_batch['lengths']}")


if __name__ == "__main__":
    main()

# 出力
# SST-2を読み込み、トークンID列に変換します。
# 語彙数: 300001
# train.tsv: 元データ数 = 67349
# train.tsv: 削除したデータ数 = 953
# train.tsv: 使用するデータ数 = 66396
# dev.tsv: 元データ数 = 872
# dev.tsv: 削除したデータ数 = 0
# dev.tsv: 使用するデータ数 = 872
# train_dataset のデータ数: 66396
# dev_dataset のデータ数: 872

# train_dataset の先頭1件:
# {'text': 'hide new secretions from the parental units ', 'input_ids': tensor([  5785,     66, 113845,     18,     12,  15095,   1594]), 'label': tensor(0)}

# train_loader から取り出した1バッチ:
# text: ['clever , ', 'hilariously , gloriously alive ', 'smart and alert , thirteen conversations about one thing is a small gem . ', 'susan sarandon , dustin hoffman and holly hunter , ']
# input_ids の形: torch.Size([4, 10])
# label: tensor([1, 1, 1, 1])
# lengths: tensor([ 1,  3, 10,  2])

# dev_loader から取り出した1バッチ:
# text: ["it 's a charming and often affecting journey . ", 'unflinchingly bleak and desperate ', 'allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker . ', "the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales . "]
# input_ids の形: torch.Size([4, 14])
# label: tensor([1, 0, 1, 1])
# lengths: tensor([ 5,  3, 14, 14])
