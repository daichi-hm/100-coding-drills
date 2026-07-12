from pathlib import Path
import csv

import torch
from transformers import BertTokenizer


MODEL_NAME = "bert-base-uncased"

# 現在のファイル 86.py があるディレクトリを取得する
CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = CURRENT_DIR.parent / "chapter7" / "SST-2"

# 訓練データのパス
TRAIN_PATH = DATA_DIR / "train.tsv"

# BERT用のtokenizerを読み込む
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def load_sst2(path, tokenizer):
    """SST-2のtsvファイルを読み込み、文・ラベル・input_idsを返す関数"""

    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    examples = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            sentence = row["sentence"]
            label = int(row["label"])

            # BERTに入力するためのID列を作る
            # add_special_tokens=True により、
            # 先頭に [CLS]、末尾に [SEP] が付く
            input_ids = tokenizer.encode(
                sentence,
                add_special_tokens=True,
            )

            examples.append(
                {
                    "sentence": sentence,
                    "label": label,
                    "input_ids": input_ids,
                }
            )

    return examples


def create_batch(examples, tokenizer):
    """複数の事例から、BERTに入力できるミニバッチを作る関数"""

    # 事例ごとのinput_idsを取り出す
    input_ids_list = [example["input_ids"] for example in examples]

    # 事例ごとのラベルを取り出す
    labels = [example["label"] for example in examples]

    # ミニバッチ内で一番長い系列長を調べる
    max_length = max(len(input_ids) for input_ids in input_ids_list)

    # paddingに使うID
    # BERTでは [PAD] に対応するIDを使う
    pad_token_id = tokenizer.pad_token_id

    padded_input_ids = []
    attention_masks = []

    for input_ids in input_ids_list:
        # 足りない長さを計算する
        padding_length = max_length - len(input_ids)

        # input_idsの後ろに [PAD] のIDを追加して長さを揃える
        padded_ids = input_ids + [pad_token_id] * padding_length

        # attention_maskを作る
        # 実際のトークンは1、[PAD]は0
        attention_mask = [1] * len(input_ids) + [0] * padding_length

        padded_input_ids.append(padded_ids)
        attention_masks.append(attention_mask)

    # PythonのリストをPyTorchのTensorに変換する
    batch = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

    return batch


# 85番と同じようにSST-2の訓練データを読み込む
train_data = load_sst2(TRAIN_PATH, tokenizer)

# 問題文に合わせて、訓練データの冒頭4事例を使う
small_examples = train_data[:4]

# ミニバッチを作成する
batch = create_batch(small_examples, tokenizer)

# 確認用に元の文・ラベルを表示する
print("examples:")
for i, example in enumerate(small_examples):
    print(f"\n--- example {i} ---")
    print("sentence:", example["sentence"])
    print("label:", example["label"])
    print("tokens:", tokenizer.convert_ids_to_tokens(example["input_ids"]))
    print("length:", len(example["input_ids"]))

# ミニバッチの中身を表示する
print("\nmini-batch:")
print("input_ids:")
print(batch["input_ids"])

print("\nattention_mask:")
print(batch["attention_mask"])

print("\nlabels:")
print(batch["labels"])

print("\nshapes:")
print("input_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
print("labels shape:", batch["labels"].shape)

# padding後のトークン列も確認する
print("\npadded tokens:")
for i, input_ids in enumerate(batch["input_ids"]):
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    print(f"{i}: {tokens}")

#出力
# examples:

# --- example 0 ---
# sentence: hide new secretions from the parental units 
# label: 0
# tokens: ['[CLS]', 'hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units', '[SEP]']
# length: 10

# --- example 1 ---
# sentence: contains no wit , only labored gags 
# label: 0
# tokens: ['[CLS]', 'contains', 'no', 'wit', ',', 'only', 'labor', '##ed', 'gag', '##s', '[SEP]']
# length: 11

# --- example 2 ---
# sentence: that loves its characters and communicates something rather beautiful about human nature 
# label: 1
# tokens: ['[CLS]', 'that', 'loves', 'its', 'characters', 'and', 'communicate', '##s', 'something', 'rather', 'beautiful', 'about', 'human', 'nature', '[SEP]']
# length: 15

# --- example 3 ---
# sentence: remains utterly satisfied to remain the same throughout 
# label: 0
# tokens: ['[CLS]', 'remains', 'utterly', 'satisfied', 'to', 'remain', 'the', 'same', 'throughout', '[SEP]']
# length: 10

# mini-batch:
# input_ids:
# tensor([[  101,  5342,  2047,  3595,  8496,  2013,  1996, 18643,  3197,   102,
#              0,     0,     0,     0,     0],
#         [  101,  3397,  2053, 15966,  1010,  2069,  4450,  2098, 18201,  2015,
#            102,     0,     0,     0,     0],
#         [  101,  2008,  7459,  2049,  3494,  1998, 10639,  2015,  2242,  2738,
#           3376,  2055,  2529,  3267,   102],
#         [  101,  3464, 12580,  8510,  2000,  3961,  1996,  2168,  2802,   102,
#              0,     0,     0,     0,     0]])

# attention_mask:
# tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

# labels:
# tensor([0, 0, 1, 0])

# shapes:
# input_ids shape: torch.Size([4, 15])
# attention_mask shape: torch.Size([4, 15])
# labels shape: torch.Size([4])

# padded tokens:
# 0: ['[CLS]', 'hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
# 1: ['[CLS]', 'contains', 'no', 'wit', ',', 'only', 'labor', '##ed', 'gag', '##s', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
# 2: ['[CLS]', 'that', 'loves', 'its', 'characters', 'and', 'communicate', '##s', 'something', 'rather', 'beautiful', 'about', 'human', 'nature', '[SEP]']
# 3: ['[CLS]', 'remains', 'utterly', 'satisfied', 'to', 'remain', 'the', 'same', 'throughout', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']