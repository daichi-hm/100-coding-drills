from pathlib import Path
import csv
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# =========================
# パス設定
# =========================

# このファイルが置かれているディレクトリ
# 例: nlp100/chapter8
BASE_DIR = Path(__file__).resolve().parent

# プロジェクト全体のルートディレクトリ
PROJECT_ROOT = BASE_DIR.parent

# 70番で作成した単語埋め込みデータ
EMBEDDING_FILE = BASE_DIR / "data" / "embeddings.pt"

# SST-2のtrain/devデータ
TRAIN_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "train.tsv"
DEV_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "dev.tsv"

# 79番で学習したモデルの保存先
MODEL_OUTPUT_FILE = BASE_DIR / "data" / "mlp_bow_model_79.pt"


# =========================
# 学習設定
# =========================

# GPUが使えるならGPUを使う。
# CUDA_VISIBLE_DEVICES=1 のように指定すれば、そのGPUだけが使われる。
USE_GPU = True

# 何件ずつまとめて学習するか
BATCH_SIZE = 64

# 何周学習するか
NUM_EPOCHS = 5

# 学習率
LEARNING_RATE = 1e-3

# 過学習を少し抑えるための重み減衰
WEIGHT_DECAY = 1e-4

# 何ステップごとに途中経過を表示するか
LOG_INTERVAL = 100

# =========================
# 79番で追加するMLPの設定
# =========================

# 隠れ層の次元数
#
# 78番までのBoWモデルでは、
#   300次元 → 2クラス
# に直接変換していた。
#
# 79番では、
#   300次元 → 100次元 → 2クラス
# という多層ニューラルネットワークにする。
HIDDEN_DIM = 100

# Dropoutの割合
#
# Dropoutは、学習中に一部の値をランダムに0にすることで、
# 訓練データに合わせすぎることを防ぐ。
DROPOUT_RATE = 0.5


def tokenize(text):
    """
    文をトークン列に変換する関数。

    SST-2はある程度スペース区切りされているので、
    小文字化してsplitする。
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
        label = torch.tensor(example["label"], dtype=torch.long)

        return {
            "text": example["text"],
            "input_ids": input_ids,
            "label": label,
        }


def collate_batch(batch):
    """
    DataLoaderで複数事例を1つのミニバッチにまとめる関数。

    75番の仕様と同じく、
    1. 長い順に並び替える
    2. 一番長い文に合わせて0でpaddingする
    3. input_ids, label, lengths を返す
    """

    sorted_batch = sorted(
        batch,
        key=lambda example: len(example["input_ids"]),
        reverse=True,
    )

    texts = []
    input_ids_list = []
    labels = []
    lengths = []

    for example in sorted_batch:
        texts.append(example["text"])
        input_ids_list.append(example["input_ids"])
        labels.append(example["label"])
        lengths.append(len(example["input_ids"]))

    padded_input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0,
    )

    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return {
        "text": texts,
        "input_ids": padded_input_ids,
        "label": labels,
        "lengths": lengths,
    }


class MLPBoWClassifier(nn.Module):
    """
    多層ニューラルネットワークを使ったBoW分類モデル。

    78番までの単純なBoWモデル:
        単語ベクトル平均
        ↓
        Linear(300 → 2)

    79番のモデル:
        単語ベクトル平均
        ↓
        Linear(300 → hidden_dim)
        ↓
        ReLU
        ↓
        Dropout
        ↓
        Linear(hidden_dim → 2)

    つまり、文ベクトルを直接2クラスに変換するのではなく、
    一度隠れ層を通してから分類する。
    """

    def __init__(
        self,
        embedding_matrix,
        hidden_dim=100,
        num_classes=2,
        dropout_rate=0.5,
    ):
        super().__init__()

        # 70番で作成した単語埋め込み行列を使う。
        #
        # 79番では、アーキテクチャ変更に集中するため、
        # 単語埋め込みは固定する。
        #
        # freeze=True:
        #   embedding.weight は学習で更新されない。
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=True,
            padding_idx=0,
        )

        embedding_dim = embedding_matrix.size(1)

        # =========================
        # 79番で追加する隠れ層
        # =========================
        #
        # 300次元の文ベクトルを hidden_dim 次元に変換する。
        self.hidden = nn.Linear(embedding_dim, hidden_dim)

        # ReLUは、負の値を0にする非線形関数。
        #
        # Linearだけを2回重ねても、実質的には1つのLinearにまとめられてしまう。
        # そのため、途中にReLUを入れて、表現力を上げる。
        self.relu = nn.ReLU()

        # Dropoutは過学習を防ぐための層。
        self.dropout = nn.Dropout(dropout_rate)

        # hidden_dim次元の特徴から、2クラスのスコアを出す。
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        """
        順伝播。

        input_ids:
            [バッチサイズ, 文長]

        lengths:
            [バッチサイズ]

        戻り値:
            logits
            [バッチサイズ, 2]
        """

        # input_ids:
        #   [バッチサイズ, 文長]
        #
        # embeddings:
        #   [バッチサイズ, 文長, 300]
        embeddings = self.embedding(input_ids)

        # 文ごとに単語ベクトルを足す。
        #
        # summed_vectors:
        #   [バッチサイズ, 300]
        summed_vectors = embeddings.sum(dim=1)

        # lengths:
        #   [バッチサイズ]
        #   ↓
        #   [バッチサイズ, 1]
        #
        # 割り算できる形にする。
        lengths = lengths.clamp(min=1).unsqueeze(1).to(summed_vectors.dtype)

        # 単語ベクトルの平均を取る。
        #
        # sentence_vectors:
        #   [バッチサイズ, 300]
        sentence_vectors = summed_vectors / lengths

        # =========================
        # 79番で追加したMLP部分
        # =========================

        # 300次元 → hidden_dim次元
        hidden_vectors = self.hidden(sentence_vectors)

        # 非線形変換
        hidden_vectors = self.relu(hidden_vectors)

        # 学習時だけ一部の値を0にする
        hidden_vectors = self.dropout(hidden_vectors)

        # hidden_dim次元 → 2クラス
        logits = self.output(hidden_vectors)

        return logits


def evaluate(model, data_loader, criterion, device):
    """
    指定されたデータに対して、lossとaccuracyを計算する関数。

    ここでは評価だけ行い、重み更新はしない。
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, lengths)

            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predicted_labels = torch.argmax(logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()

            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def main():
    print("79番: BoWモデルを多層ニューラルネットワークに変更して学習します。")

    # =========================
    # deviceを決める
    # =========================

    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用デバイス: {device}")
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"使用デバイス: {device}")

    # =========================
    # 70番の単語埋め込みデータを読み込む
    # =========================

    data = torch.load(EMBEDDING_FILE, map_location="cpu")

    embedding_matrix = data["embedding_matrix"]
    token_to_id = data["token_to_id"]

    print(f"embedding_matrix の形: {embedding_matrix.shape}")
    print(f"語彙数: {len(token_to_id)}")

    # =========================
    # train.tsv と dev.tsv を読み込む
    # =========================

    train_examples = load_sst_file(TRAIN_FILE, token_to_id)
    dev_examples = load_sst_file(DEV_FILE, token_to_id)

    train_dataset = SSTDataset(train_examples)
    dev_dataset = SSTDataset(dev_examples)

    # =========================
    # DataLoaderを作る
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

    print(f"train_dataset のデータ数: {len(train_dataset)}")
    print(f"dev_dataset のデータ数: {len(dev_dataset)}")
    print(f"バッチサイズ: {BATCH_SIZE}")
    print(f"1エポックあたりのtrainバッチ数: {len(train_loader)}")
    print(f"devバッチ数: {len(dev_loader)}")

    # =========================
    # モデルを作る
    # =========================

    model = MLPBoWClassifier(
        embedding_matrix=embedding_matrix,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        dropout_rate=DROPOUT_RATE,
    )

    model = model.to(device)

    print(model)

    # =========================
    # 学習対象パラメータを確認する
    # =========================

    total_params = 0
    trainable_params = 0

    for parameter in model.parameters():
        total_params += parameter.numel()

        if parameter.requires_grad:
            trainable_params += parameter.numel()

    print(f"全パラメータ数: {total_params}")
    print(f"学習対象パラメータ数: {trainable_params}")

    print(f"embedding.weight requires_grad: {model.embedding.weight.requires_grad}")
    print(f"hidden.weight requires_grad: {model.hidden.weight.requires_grad}")
    print(f"output.weight requires_grad: {model.output.weight.requires_grad}")

    # =========================
    # 損失関数と最適化手法を用意する
    # =========================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # =========================
    # 学習ループ
    # =========================

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_start_time = time.time()

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, lengths)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            # =========================
            # ログ表示用の値を集計する
            # =========================

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predicted_labels = torch.argmax(logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()

            total_examples += batch_size

            if step % LOG_INTERVAL == 0:
                average_loss = total_loss / total_examples
                accuracy = total_correct / total_examples

                print(
                    f"epoch {epoch} | "
                    f"step {step}/{len(train_loader)} | "
                    f"train loss {average_loss:.4f} | "
                    f"train accuracy {accuracy:.4f}"
                )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.time() - epoch_start_time

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples

        dev_loss, dev_accuracy = evaluate(
            model=model,
            data_loader=dev_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"[epoch {epoch} finished] "
            f"train loss {train_loss:.4f} | "
            f"train accuracy {train_accuracy:.4f} | "
            f"dev loss {dev_loss:.4f} | "
            f"dev accuracy {dev_accuracy:.4f} | "
            f"time {epoch_time:.2f} sec"
        )

    # =========================
    # 学習済みモデルを保存する
    # =========================

    MODEL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # embedding層は固定しているため、ここでは学習されたMLP部分だけ保存する。
    #
    # 保存対象:
    #   hidden層
    #   output層
    #
    # embeddings.ptは70番で保存済みなので、ここでは保存しない。
    torch.save(
        {
            "hidden_state_dict": {
                key: value.cpu()
                for key, value in model.hidden.state_dict().items()
            },
            "output_state_dict": {
                key: value.cpu()
                for key, value in model.output.state_dict().items()
            },
            "model_type": "MLPBoWClassifier",
            "embedding_dim": embedding_matrix.size(1),
            "hidden_dim": HIDDEN_DIM,
            "num_classes": 2,
            "dropout_rate": DROPOUT_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        MODEL_OUTPUT_FILE,
    )

    print("学習済みモデルを保存しました")


if __name__ == "__main__":
    main()

#出力
# 79番: BoWモデルを多層ニューラルネットワークに変更して学習します。
# 使用デバイス: cuda
# GPU名: NVIDIA RTX A6000
# embedding_matrix の形: torch.Size([300001, 300])
# 語彙数: 300001
# train.tsv: 元データ数 = 67349
# train.tsv: 削除したデータ数 = 953
# train.tsv: 使用するデータ数 = 66396
# dev.tsv: 元データ数 = 872
# dev.tsv: 削除したデータ数 = 0
# dev.tsv: 使用するデータ数 = 872
# train_dataset のデータ数: 66396
# dev_dataset のデータ数: 872
# バッチサイズ: 64
# 1エポックあたりのtrainバッチ数: 1038
# devバッチ数: 14
# MLPBoWClassifier(
#   (embedding): Embedding(300001, 300, padding_idx=0)
#   (hidden): Linear(in_features=300, out_features=100, bias=True)
#   (relu): ReLU()
#   (dropout): Dropout(p=0.5, inplace=False)
#   (output): Linear(in_features=100, out_features=2, bias=True)
# )
# 全パラメータ数: 90030602
# 学習対象パラメータ数: 30302
# embedding.weight requires_grad: False
# hidden.weight requires_grad: True
# output.weight requires_grad: True
# epoch 1 | step 100/1038 | train loss 0.5487 | train accuracy 0.7292
# epoch 1 | step 200/1038 | train loss 0.4730 | train accuracy 0.7760
# epoch 1 | step 300/1038 | train loss 0.4435 | train accuracy 0.7923
# epoch 1 | step 400/1038 | train loss 0.4278 | train accuracy 0.8035
# epoch 1 | step 500/1038 | train loss 0.4164 | train accuracy 0.8107
# epoch 1 | step 600/1038 | train loss 0.4083 | train accuracy 0.8152
# epoch 1 | step 700/1038 | train loss 0.4017 | train accuracy 0.8187
# epoch 1 | step 800/1038 | train loss 0.3980 | train accuracy 0.8202
# epoch 1 | step 900/1038 | train loss 0.3957 | train accuracy 0.8219
# epoch 1 | step 1000/1038 | train loss 0.3932 | train accuracy 0.8236
# [epoch 1 finished] train loss 0.3921 | train accuracy 0.8239 | dev loss 0.4316 | dev accuracy 0.8016 | time 1.95 sec
# epoch 2 | step 100/1038 | train loss 0.3506 | train accuracy 0.8436
# epoch 2 | step 200/1038 | train loss 0.3530 | train accuracy 0.8438
# epoch 2 | step 300/1038 | train loss 0.3514 | train accuracy 0.8438
# epoch 2 | step 400/1038 | train loss 0.3505 | train accuracy 0.8450
# epoch 2 | step 500/1038 | train loss 0.3530 | train accuracy 0.8438
# epoch 2 | step 600/1038 | train loss 0.3538 | train accuracy 0.8439
# epoch 2 | step 700/1038 | train loss 0.3549 | train accuracy 0.8438
# epoch 2 | step 800/1038 | train loss 0.3548 | train accuracy 0.8440
# epoch 2 | step 900/1038 | train loss 0.3551 | train accuracy 0.8441
# epoch 2 | step 1000/1038 | train loss 0.3548 | train accuracy 0.8441
# [epoch 2 finished] train loss 0.3547 | train accuracy 0.8439 | dev loss 0.4289 | dev accuracy 0.8108 | time 2.66 sec
# epoch 3 | step 100/1038 | train loss 0.3595 | train accuracy 0.8452
# epoch 3 | step 200/1038 | train loss 0.3560 | train accuracy 0.8445
# epoch 3 | step 300/1038 | train loss 0.3538 | train accuracy 0.8439
# epoch 3 | step 400/1038 | train loss 0.3489 | train accuracy 0.8459
# epoch 3 | step 500/1038 | train loss 0.3481 | train accuracy 0.8480
# epoch 3 | step 600/1038 | train loss 0.3472 | train accuracy 0.8485
# epoch 3 | step 700/1038 | train loss 0.3464 | train accuracy 0.8495
# epoch 3 | step 800/1038 | train loss 0.3462 | train accuracy 0.8494
# epoch 3 | step 900/1038 | train loss 0.3461 | train accuracy 0.8490
# epoch 3 | step 1000/1038 | train loss 0.3473 | train accuracy 0.8486
# [epoch 3 finished] train loss 0.3470 | train accuracy 0.8486 | dev loss 0.4361 | dev accuracy 0.8062 | time 2.64 sec
# epoch 4 | step 100/1038 | train loss 0.3347 | train accuracy 0.8542
# epoch 4 | step 200/1038 | train loss 0.3351 | train accuracy 0.8562
# epoch 4 | step 300/1038 | train loss 0.3365 | train accuracy 0.8558
# epoch 4 | step 400/1038 | train loss 0.3363 | train accuracy 0.8554
# epoch 4 | step 500/1038 | train loss 0.3378 | train accuracy 0.8541
# epoch 4 | step 600/1038 | train loss 0.3387 | train accuracy 0.8535
# epoch 4 | step 700/1038 | train loss 0.3381 | train accuracy 0.8539
# epoch 4 | step 800/1038 | train loss 0.3394 | train accuracy 0.8532
# epoch 4 | step 900/1038 | train loss 0.3400 | train accuracy 0.8530
# epoch 4 | step 1000/1038 | train loss 0.3398 | train accuracy 0.8529
# [epoch 4 finished] train loss 0.3398 | train accuracy 0.8532 | dev loss 0.4221 | dev accuracy 0.8085 | time 1.79 sec
# epoch 5 | step 100/1038 | train loss 0.3377 | train accuracy 0.8502
# epoch 5 | step 200/1038 | train loss 0.3364 | train accuracy 0.8522
# epoch 5 | step 300/1038 | train loss 0.3359 | train accuracy 0.8529
# epoch 5 | step 400/1038 | train loss 0.3330 | train accuracy 0.8542
# epoch 5 | step 500/1038 | train loss 0.3337 | train accuracy 0.8543
# epoch 5 | step 600/1038 | train loss 0.3328 | train accuracy 0.8550
# epoch 5 | step 700/1038 | train loss 0.3327 | train accuracy 0.8554
# epoch 5 | step 800/1038 | train loss 0.3320 | train accuracy 0.8561
# epoch 5 | step 900/1038 | train loss 0.3331 | train accuracy 0.8557
# epoch 5 | step 1000/1038 | train loss 0.3336 | train accuracy 0.8554
# [epoch 5 finished] train loss 0.3342 | train accuracy 0.8554 | dev loss 0.4241 | dev accuracy 0.8165 | time 2.56 sec
# 学習済みモデルを保存しました