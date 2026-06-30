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

# 71番で使ったSST-2のtrain.tsv
TRAIN_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "train.tsv"

# 71番で使ったSST-2のdev.tsv
DEV_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "dev.tsv"

# 77番で学習したモデルの保存先
MODEL_OUTPUT_FILE = BASE_DIR / "data" / "bow_model_77.pt"


# =========================
# 学習設定
# =========================

# 77番ではGPU上で学習する。
# GPUが使えない場合は、問題77の条件を満たせないのでエラーにする。
REQUIRE_GPU = True

# 学習データを何件ずつまとめて処理するか
BATCH_SIZE = 64

# 何周学習するか
NUM_EPOCHS = 5

# 学習率
LEARNING_RATE = 1e-3

# 何ステップごとに途中経過を表示するか
LOG_INTERVAL = 100


def tokenize(text):
    """
    文をトークン列に変換する関数。

    SST-2がすでにある程度スペース区切りされているので、
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

    75番の仕様:
    1. トークン列の長さが長い順に事例を並び替える
    2. 一番長い系列長に合わせて、短い文を0でpaddingする
    3. input_ids, label, lengths をテンソルとして返す
    """

    # 長さが長い順に並び替える
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

    # 長さの違うinput_idsを0でpaddingする
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


class BoWClassifier(nn.Module):
    """
    Bag of Words によるポジネガ分類モデル。

    1. input_idsを単語ベクトル列に変換する
    2. 単語ベクトルの平均を取って文ベクトルを作る
    3. 文ベクトルをlinear層に入れて2クラスのスコアを出す
    """

    def __init__(self, embedding_matrix, num_classes=2):
        super().__init__()

        # freeze=True なので、単語埋め込み行列は学習中に更新しない。
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=True,
            padding_idx=0,
        )

        embedding_dim = embedding_matrix.size(1)

        self.linear = nn.Linear(embedding_dim, num_classes)

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

        # input_idsを単語ベクトルに変換する
        embeddings = self.embedding(input_ids)

        # 文長方向に単語ベクトルを足す
        summed_vectors = embeddings.sum(dim=1)

        # 各文の長さで割るために、
        # lengthsを [バッチサイズ] から [バッチサイズ, 1] にする
        lengths = lengths.clamp(min=1).unsqueeze(1).to(summed_vectors.dtype)

        # 単語ベクトルの平均を取る
        sentence_vectors = summed_vectors / lengths

        # 2クラスのスコアを出す
        logits = self.linear(sentence_vectors)

        return logits


def get_device():
    """
    77番用にGPUを使うdeviceを決める関数。

    GPUが使える場合:
        cuda を返す。

    GPUが使えない場合:
        REQUIRE_GPU=True ならエラーにする。
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用デバイス: {device}")
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        return device

    if REQUIRE_GPU:
        raise RuntimeError(
            "GPUが使えません。77番はGPU上での学習が条件なので、"
            "AIXサーバなどGPUが使える環境で実行してください。"
        )

    device = torch.device("cpu")
    print(f"使用デバイス: {device}")
    return device


def evaluate(model, data_loader, criterion, device):
    """
    指定されたデータに対して、lossとaccuracyを計算する関数。

    ここでは学習はしない。
    つまり、重み更新はしない。
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            # GPU学習なので、評価時もテンソルをGPUへ送る。
            #
            # non_blocking=True は、pin_memory=True と組み合わせると、
            # CPUからGPUへの転送を少し効率化できることがある。
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

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
    print("76番のモデル学習をGPU上で実行します。")

    # =========================
    # deviceを決める
    # =========================

    device = get_device()

    # =========================
    # 70番の単語埋め込みデータを読み込む
    # =========================

    # まずCPU上に読み込む。
    # その後 model.to(device) でモデル全体をGPUへ移す。
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
        pin_memory=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    print(f"train_dataset のデータ数: {len(train_dataset)}")
    print(f"dev_dataset のデータ数: {len(dev_dataset)}")
    print(f"バッチサイズ: {BATCH_SIZE}")
    print(f"1エポックあたりのtrainバッチ数: {len(train_loader)}")
    print(f"devバッチ数: {len(dev_loader)}")

    # =========================
    # モデルを作る
    # =========================

    model = BoWClassifier(embedding_matrix)

    # モデルをGPUへ移動する。
    #
    # ここで embedding層とlinear層がGPUに移る。
    # これをしないと、GPU上のinput_idsとCPU上のmodelで計算しようとしてエラーになる。
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

    # =========================
    # 損失関数と最適化手法を用意する
    # =========================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # =========================
    # 学習ループ
    # =========================

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        # GPU処理は非同期で進むことがあるので、
        # 時間を測る前にGPU処理の完了を待つ。
        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_start_time = time.time()

        for step, batch in enumerate(train_loader, start=1):
            # DataLoaderから取り出したテンソルをGPUへ移す。
            #
            # 76番ではCPUに送っていたが、77番ではGPUに送る。
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

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

        # GPU処理は非同期で進むことがあるので、
        # 時間を測る前にGPU処理の完了を待つ。
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

    # 保存するときはCPUに戻してから保存すると、
    # GPUがない環境でも読み込みやすい。
    linear_state_dict = {
        key: value.cpu()
        for key, value in model.linear.state_dict().items()
    }

    torch.save(
        {
            "linear_state_dict": linear_state_dict,
            "num_classes": 2,
            "embedding_dim": embedding_matrix.size(1),
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "device": str(device),
        },
        MODEL_OUTPUT_FILE,
    )

    print("学習済みモデルを保存しました")


if __name__ == "__main__":
    main()

#出力
# 76番のモデル学習をGPU上で実行します。
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
# BoWClassifier(
#   (embedding): Embedding(300001, 300, padding_idx=0)
#   (linear): Linear(in_features=300, out_features=2, bias=True)
# )
# 全パラメータ数: 90000902
# 学習対象パラメータ数: 602
# epoch 1 | step 100/1038 | train loss 0.6298 | train accuracy 0.6558
# epoch 1 | step 200/1038 | train loss 0.5922 | train accuracy 0.7262
# epoch 1 | step 300/1038 | train loss 0.5645 | train accuracy 0.7523
# epoch 1 | step 400/1038 | train loss 0.5458 | train accuracy 0.7661
# epoch 1 | step 500/1038 | train loss 0.5295 | train accuracy 0.7757
# epoch 1 | step 600/1038 | train loss 0.5168 | train accuracy 0.7831
# epoch 1 | step 700/1038 | train loss 0.5056 | train accuracy 0.7894
# epoch 1 | step 800/1038 | train loss 0.4969 | train accuracy 0.7935
# epoch 1 | step 900/1038 | train loss 0.4892 | train accuracy 0.7968
# epoch 1 | step 1000/1038 | train loss 0.4830 | train accuracy 0.7989
# [epoch 1 finished] train loss 0.4801 | train accuracy 0.8004 | dev loss 0.5052 | dev accuracy 0.7764 | time 2.49 sec
# epoch 2 | step 100/1038 | train loss 0.4065 | train accuracy 0.8309
# epoch 2 | step 200/1038 | train loss 0.4126 | train accuracy 0.8264
# epoch 2 | step 300/1038 | train loss 0.4123 | train accuracy 0.8254
# epoch 2 | step 400/1038 | train loss 0.4097 | train accuracy 0.8282
# epoch 2 | step 500/1038 | train loss 0.4097 | train accuracy 0.8281
# epoch 2 | step 600/1038 | train loss 0.4078 | train accuracy 0.8289
# epoch 2 | step 700/1038 | train loss 0.4072 | train accuracy 0.8289
# epoch 2 | step 800/1038 | train loss 0.4054 | train accuracy 0.8297
# epoch 2 | step 900/1038 | train loss 0.4025 | train accuracy 0.8307
# epoch 2 | step 1000/1038 | train loss 0.3997 | train accuracy 0.8321
# [epoch 2 finished] train loss 0.3995 | train accuracy 0.8320 | dev loss 0.4768 | dev accuracy 0.7856 | time 1.54 sec
# epoch 3 | step 100/1038 | train loss 0.3845 | train accuracy 0.8409
# epoch 3 | step 200/1038 | train loss 0.3834 | train accuracy 0.8400
# epoch 3 | step 300/1038 | train loss 0.3844 | train accuracy 0.8382
# epoch 3 | step 400/1038 | train loss 0.3859 | train accuracy 0.8376
# epoch 3 | step 500/1038 | train loss 0.3864 | train accuracy 0.8360
# epoch 3 | step 600/1038 | train loss 0.3860 | train accuracy 0.8362
# epoch 3 | step 700/1038 | train loss 0.3852 | train accuracy 0.8367
# epoch 3 | step 800/1038 | train loss 0.3848 | train accuracy 0.8373
# epoch 3 | step 900/1038 | train loss 0.3846 | train accuracy 0.8370
# epoch 3 | step 1000/1038 | train loss 0.3843 | train accuracy 0.8373
# [epoch 3 finished] train loss 0.3844 | train accuracy 0.8374 | dev loss 0.4661 | dev accuracy 0.7867 | time 1.57 sec
# epoch 4 | step 100/1038 | train loss 0.3855 | train accuracy 0.8375
# epoch 4 | step 200/1038 | train loss 0.3847 | train accuracy 0.8379
# epoch 4 | step 300/1038 | train loss 0.3845 | train accuracy 0.8378
# epoch 4 | step 400/1038 | train loss 0.3808 | train accuracy 0.8400
# epoch 4 | step 500/1038 | train loss 0.3778 | train accuracy 0.8412
# epoch 4 | step 600/1038 | train loss 0.3770 | train accuracy 0.8417
# epoch 4 | step 700/1038 | train loss 0.3779 | train accuracy 0.8409
# epoch 4 | step 800/1038 | train loss 0.3782 | train accuracy 0.8398
# epoch 4 | step 900/1038 | train loss 0.3783 | train accuracy 0.8398
# epoch 4 | step 1000/1038 | train loss 0.3777 | train accuracy 0.8402
# [epoch 4 finished] train loss 0.3782 | train accuracy 0.8399 | dev loss 0.4597 | dev accuracy 0.7913 | time 2.21 sec
# epoch 5 | step 100/1038 | train loss 0.3792 | train accuracy 0.8373
# epoch 5 | step 200/1038 | train loss 0.3833 | train accuracy 0.8334
# epoch 5 | step 300/1038 | train loss 0.3795 | train accuracy 0.8364
# epoch 5 | step 400/1038 | train loss 0.3761 | train accuracy 0.8388
# epoch 5 | step 500/1038 | train loss 0.3755 | train accuracy 0.8397
# epoch 5 | step 600/1038 | train loss 0.3745 | train accuracy 0.8413
# epoch 5 | step 700/1038 | train loss 0.3758 | train accuracy 0.8404
# epoch 5 | step 800/1038 | train loss 0.3777 | train accuracy 0.8400
# epoch 5 | step 900/1038 | train loss 0.3761 | train accuracy 0.8405
# epoch 5 | step 1000/1038 | train loss 0.3752 | train accuracy 0.8406
# [epoch 5 finished] train loss 0.3751 | train accuracy 0.8408 | dev loss 0.4561 | dev accuracy 0.7947 | time 2.27 sec
# 学習済みモデルを保存しました