from pathlib import Path
import csv
import copy

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_NAME = "bert-base-uncased"

# ハイパーパラメータ
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# 現在のファイル 87.py があるディレクトリを取得する
CURRENT_DIR = Path(__file__).resolve().parent

DATA_DIR = CURRENT_DIR.parent / "chapter7" / "SST-2"

TRAIN_PATH = DATA_DIR / "train.tsv"
DEV_PATH = DATA_DIR / "dev.tsv"


def load_sst2(path):
    """SST-2のtsvファイルを読み込み、文とラベルを返す関数"""

    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    sentences = []
    labels = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            sentence = row["sentence"]
            label = int(row["label"])

            sentences.append(sentence)
            labels.append(label)

    return sentences, labels


class SST2Dataset(Dataset):
    """SST-2をPyTorchのDatasetとして扱うためのクラス"""

    def __init__(self, sentences, labels, tokenizer, max_length):
        # 文をまとめてBERTに入力できる形式に変換する
        # padding=True:
        #   バッチ化できるように長さを揃える
        # truncation=True:
        #   長すぎる文はmax_lengthで切る
        # return_tensors="pt":
        #   PyTorchのTensorとして返す
        self.encodings = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # ラベルもTensorにする
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        # データ数を返す
        return len(self.labels)

    def __getitem__(self, idx):
        # idx番目のデータを1件返す
        item = {
            key: value[idx]
            for key, value in self.encodings.items()
        }

        # 正解ラベルを追加する
        # BertForSequenceClassificationは "labels" という名前で渡すと、
        # 自動でlossを計算してくれる
        item["labels"] = self.labels[idx]

        return item


def train_one_epoch(model, data_loader, optimizer, device):
    """1 epoch分、訓練データでモデルを学習する関数"""

    # 学習モードにする
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in data_loader:
        # batchの中身をGPUまたはCPUに送る
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # 前回の勾配をリセットする
        optimizer.zero_grad()

        # BERTに入力する
        # labelsも渡しているので、outputs.lossが自動で計算される
        outputs = model(**batch)

        #予測がどれくらい間違っていたかを表す値
        loss = outputs.loss
        #logitsは、各クラスのスコア
        logits = outputs.logits

        # 誤差逆伝播
        loss.backward()

        # パラメータ更新
        optimizer.step()

        # lossを記録する
        total_loss += loss.item()

        # 予測ラベルを求める
        predictions = torch.argmax(logits, dim=1)

        # 正解ラベル
        labels = batch["labels"]

        # 正解数を数える
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def evaluate(model, data_loader, device):
    """検証データでlossとaccuracyを計算する関数"""

    # 評価モードにする
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # 評価時は勾配計算をしない
    with torch.no_grad():
        for batch in data_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            labels = batch["labels"]

            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def main():
    # GPUが使えるならGPUを使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # tokenizerを読み込む
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # SST-2を読み込む
    train_sentences, train_labels = load_sst2(TRAIN_PATH)
    dev_sentences, dev_labels = load_sst2(DEV_PATH)

    print("train size:", len(train_sentences))
    print("dev size:", len(dev_sentences))

    # Datasetを作る
    train_dataset = SST2Dataset(
        train_sentences,
        train_labels,
        tokenizer,
        MAX_LENGTH,
    )

    dev_dataset = SST2Dataset(
        dev_sentences,
        dev_labels,
        tokenizer,
        MAX_LENGTH,
    )

    # DataLoaderを作る
    # shuffle=Trueにすると、epochごとに訓練データの順番をランダムにする
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # BERTの文分類モデルを読み込む
    # num_labels=2 は、negative / positive の2クラス分類という意味
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # 最後のepochではなく、全epoch中で最も良いvalid accuracyを保存する
    best_valid_accuracy = 0.0
    best_valid_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== epoch {epoch} / {NUM_EPOCHS} =====")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        valid_loss, valid_accuracy = evaluate(
            model,
            dev_loader,
            device,
        )

        print(f"train loss: {train_loss:.6f}")
        print(f"train accuracy: {train_accuracy:.6f}")
        print(f"valid loss: {valid_loss:.6f}")
        print(f"valid accuracy: {valid_accuracy:.6f}")

        # valid accuracyが過去最高なら、そのモデルを保存しておく
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_valid_loss = valid_loss
            best_epoch = epoch

            # model.state_dict()は現在の重み
            # copy.deepcopyで、その時点の重みをコピーして保存する
            best_model_state = copy.deepcopy(model.state_dict())

            print("best model updated!")

    print("\n===== best result =====")
    print(f"best epoch: {best_epoch}")
    print(f"best valid accuracy: {best_valid_accuracy:.6f}")
    print(f"valid loss at best epoch: {best_valid_loss:.6f}")

    # bestモデルをファイルに保存する
    # 88番以降で使えるようにしておく
    output_path = CURRENT_DIR / "best_bert_sst2.pt"

    torch.save(
        {
            "model_state_dict": best_model_state,
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_valid_accuracy,
            "best_valid_loss": best_valid_loss,
        },
        output_path,
    )

    print(f"best model saved to: {output_path}")


if __name__ == "__main__":
    main()

#出力
# device: cuda
# train size: 67349
# dev size: 872

# ===== epoch 1 / 3 =====
# train loss: 0.202933
# train accuracy: 0.922568
# valid loss: 0.236477
# valid accuracy: 0.920872
# best model updated!

# ===== epoch 2 / 3 =====
# train loss: 0.108366
# train accuracy: 0.962895
# valid loss: 0.222731
# valid accuracy: 0.916284

# ===== epoch 3 / 3 =====
# train loss: 0.073770
# train accuracy: 0.974461
# valid loss: 0.246449
# valid accuracy: 0.917431

# ===== best result =====
# best epoch: 1
# best valid accuracy: 0.920872
# valid loss at best epoch: 0.236477