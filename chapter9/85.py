from pathlib import Path
import csv
from transformers import BertTokenizer


MODEL_NAME = "bert-base-uncased"

# 現在のファイル 85.py があるディレクトリを取得する
CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = CURRENT_DIR.parent / "chapter7" / "SST-2"

# 読み込むファイル
TRAIN_PATH = DATA_DIR / "train.tsv"
DEV_PATH = DATA_DIR / "dev.tsv"

# BERT用のtokenizerを読み込む
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def load_sst2(path: Path, tokenizer: BertTokenizer) -> list[dict]:
    """SST-2のtsvファイルを読み込み、文・ラベル・トークン列を返す関数"""

    # ファイルが存在しない場合は、分かりやすいエラーを出す
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    examples = []

    # SST-2のtrain.tsv/dev.tsvはタブ区切りファイル
    # 基本的には header として sentence, label が入っている
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            # 文を取り出す
            sentence = row["sentence"]

            # ラベルを取り出す
            # SST-2では 0 が negative, 1 が positive
            label = int(row["label"])

            # 文をBERTのWordPieceトークン列に変換する
            # これは [CLS] や [SEP] を付けない、純粋なトークン列
            tokens = tokenizer.tokenize(sentence)

            # 今後の86番以降でBERTに入力しやすいように、
            # [CLS] と [SEP] を含む入力ID列も作っておく
            input_ids = tokenizer.encode(
                sentence,
                add_special_tokens=True,
            )

            # input_idsを確認しやすいようにトークンへ戻したもの
            tokens_with_special = tokenizer.convert_ids_to_tokens(input_ids)

            examples.append(
                {
                    "sentence": sentence,
                    "label": label,
                    "tokens": tokens,
                    "tokens_with_special": tokens_with_special,
                    "input_ids": input_ids,
                }
            )

    return examples


# train.tsv と dev.tsv を読み込む
train_data = load_sst2(TRAIN_PATH, tokenizer)
dev_data = load_sst2(DEV_PATH, tokenizer)

# 読み込めた件数を表示する
print(f"train data size: {len(train_data)}")
print(f"dev data size: {len(dev_data)}")

# 確認用に、訓練データの先頭3件を表示する
print("\nfirst 3 examples:")
for i, example in enumerate(train_data[:3]):
    print(f"\n--- example {i} ---")
    print("sentence:")
    print(example["sentence"])

    print("label:")
    print(example["label"])

    print("tokens:")
    print(example["tokens"])

    print("tokens_with_special:")
    print(example["tokens_with_special"])

    print("input_ids:")
    print(example["input_ids"])

#出力
# train data size: 67349
# dev data size: 872

# first 3 examples:

# --- example 0 ---
# sentence:
# hide new secretions from the parental units 
# label:
# 0
# tokens:
# ['hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units']
# tokens_with_special:
# ['[CLS]', 'hide', 'new', 'secret', '##ions', 'from', 'the', 'parental', 'units', '[SEP]']
# input_ids:
# [101, 5342, 2047, 3595, 8496, 2013, 1996, 18643, 3197, 102]

# --- example 1 ---
# sentence:
# contains no wit , only labored gags 
# label:
# 0
# tokens:
# ['contains', 'no', 'wit', ',', 'only', 'labor', '##ed', 'gag', '##s']
# tokens_with_special:
# ['[CLS]', 'contains', 'no', 'wit', ',', 'only', 'labor', '##ed', 'gag', '##s', '[SEP]']
# input_ids:
# [101, 3397, 2053, 15966, 1010, 2069, 4450, 2098, 18201, 2015, 102]

# --- example 2 ---
# sentence:
# that loves its characters and communicates something rather beautiful about human nature 
# label:
# 1
# tokens:
# ['that', 'loves', 'its', 'characters', 'and', 'communicate', '##s', 'something', 'rather', 'beautiful', 'about', 'human', 'nature']
# tokens_with_special:
# ['[CLS]', 'that', 'loves', 'its', 'characters', 'and', 'communicate', '##s', 'something', 'rather', 'beautiful', 'about', 'human', 'nature', '[SEP]']
# input_ids:
# [101, 2008, 7459, 2049, 3494, 1998, 10639, 2015, 2242, 2738, 3376, 2055, 2529, 3267, 102]