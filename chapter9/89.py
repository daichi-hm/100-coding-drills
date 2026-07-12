from pathlib import Path
import csv
import copy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel


MODEL_NAME = "bert-base-uncased"

# ハイパーパラメータ
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
DROPOUT_RATE = 0.1

# 現在のファイル 89.py があるディレクトリを取得する
CURRENT_DIR = Path(__file__).resolve().parent

# chapter9 と chapter7 が同じ nlp100 ディレクトリの下にある想定
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
        # 文をBERTに入力できる形式へ変換する
        # tokenizerはここで、
        #   トークン化
        #   ID化
        #   [CLS] / [SEP] の追加
        #   padding
        #   attention_mask作成
        #   Tensor化
        # まで行う
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
        # idx番目のinput_ids, token_type_ids, attention_maskを取り出す
        item = {
            key: value[idx]
            for key, value in self.encodings.items()
        }

        # idx番目の正解ラベルを追加する
        item["labels"] = self.labels[idx]

        return item


class BertMeanPoolingClassifier(nn.Module):
    """BERTの各トークンベクトルを平均プーリングして分類する自作モデル"""

    def __init__(self, model_name, num_labels=2, dropout_rate=0.1):
        super().__init__()

        # BERT本体を読み込む
        # 87番のBertForSequenceClassificationではなく、
        # ここでは分類層なしのBertModelを使う
        self.bert = BertModel.from_pretrained(model_name)

        # bert-base-uncasedでは hidden_size は768
        hidden_size = self.bert.config.hidden_size

        # 過学習を少し防ぐためのDropout
        self.dropout = nn.Dropout(dropout_rate)

        # 分類層
        # 平均プーリングで作った768次元の文ベクトルを、
        # negative / positive の2スコアに変換する
        self.classifier = nn.Linear(hidden_size, num_labels)

        # 分類用の損失関数
        self.loss_fn = nn.CrossEntropyLoss()

    def mean_pooling(self, last_hidden_state, attention_mask):
        """[PAD]を除いて、各トークンベクトルの平均を取る関数"""

        # last_hidden_state の shape:
        #   [バッチサイズ, トークン数, 768]
        #
        # attention_mask の shape:
        #   [バッチサイズ, トークン数]
        #
        # attention_maskは、
        #   実トークンなら1
        #   [PAD]なら0
        # になっている

        # 掛け算できるように、attention_maskの最後に次元を1つ追加する
        # shape: [バッチサイズ, トークン数, 1]
        mask = attention_mask.unsqueeze(-1).float()

        # [PAD]の位置のベクトルを0にする
        # 実トークンの位置はそのまま残る
        masked_hidden_state = last_hidden_state * mask

        # トークン方向に足し合わせる
        # shape: [バッチサイズ, 768]
        sum_vectors = masked_hidden_state.sum(dim=1)

        # 文ごとに、実トークン数を数える
        # shape: [バッチサイズ, 1]
        token_counts = mask.sum(dim=1)

        # 0で割ることを防ぐため、最小値を小さな正の値にする
        token_counts = token_counts.clamp(min=1e-9)

        # 平均ベクトルを作る
        # shape: [バッチサイズ, 768]
        mean_vectors = sum_vectors / token_counts

        return mean_vectors

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """モデルの順伝播"""

        # BERT本体に入力する
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # BERT最終層の各トークンベクトル
        # shape: [バッチサイズ, トークン数, 768]
        last_hidden_state = outputs.last_hidden_state

        # [PAD]を除いて、各トークンベクトルの平均を取る
        # shape: [バッチサイズ, 768]
        pooled_vector = self.mean_pooling(
            last_hidden_state,
            attention_mask,
        )

        # Dropoutをかける
        pooled_vector = self.dropout(pooled_vector)

        # 分類層に通す
        # shape: [バッチサイズ, 2]
        logits = self.classifier(pooled_vector)

        # labelsがある場合だけlossを計算する
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


def train_one_epoch(model, data_loader, optimizer, device):
    """1 epoch分、訓練データでモデルを学習する関数"""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in data_loader:
        # batch内のTensorをGPUまたはCPUへ送る
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # 前回の勾配をリセットする
        optimizer.zero_grad()

        # モデルに入力する
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            labels=batch["labels"],
        )

        loss = outputs["loss"]
        logits = outputs["logits"]

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

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                labels=batch["labels"],
            )

            loss = outputs["loss"]
            logits = outputs["logits"]

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

    # 89番用の自作モデルを作る
    # 87番とは違い、BertForSequenceClassificationではなく、
    # BertModel + 平均プーリング + 自作分類層を使う
    model = BertMeanPoolingClassifier(
        model_name=MODEL_NAME,
        num_labels=2,
        dropout_rate=DROPOUT_RATE,
    )

    model.to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # 事前指示に従って、
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

        # valid accuracyが過去最高なら、その時点のモデルを保存しておく
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_valid_loss = valid_loss
            best_epoch = epoch

            # その時点の重みをメモリ上にコピーする
            best_model_state = copy.deepcopy(model.state_dict())

            print("best model updated!")

    print("\n===== best result =====")
    print(f"best epoch: {best_epoch}")
    print(f"best valid accuracy: {best_valid_accuracy:.6f}")
    print(f"valid loss at best epoch: {best_valid_loss:.6f}")

    # bestモデルを保存する
    output_path = CURRENT_DIR / "best_bert_mean_pooling_sst2.pt"

    torch.save(
        {
            "model_state_dict": best_model_state,
            "best_epoch": best_epoch,
            "best_valid_accuracy": best_valid_accuracy,
            "best_valid_loss": best_valid_loss,
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "architecture": "BertMeanPoolingClassifier",
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
# train loss: 0.200950
# train accuracy: 0.922686
# valid loss: 0.209715
# valid accuracy: 0.920872
# best model updated!

# ===== epoch 2 / 3 =====
# train loss: 0.105810
# train accuracy: 0.963652
# valid loss: 0.233043
# valid accuracy: 0.915138

# ===== epoch 3 / 3 =====
# train loss: 0.069216
# train accuracy: 0.976035
# valid loss: 0.245844
# valid accuracy: 0.928899
# best model updated!

# ===== best result =====
# best epoch: 3
# best valid accuracy: 0.928899
# valid loss at best epoch: 0.245844
# best model saved to: /data/student/h2630115/nlp100/chapter9/best_bert_mean_pooling_sst2.pt