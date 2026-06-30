from pathlib import Path
import csv
import time
import collections
import collections.abc

if not hasattr(collections, "Container"):
    collections.Container = collections.abc.Container

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

# 78番で学習したモデルの保存先
MODEL_OUTPUT_FILE = BASE_DIR / "data" / "bow_model_78.pt"


# =========================
# 学習設定
# =========================

# 78番ではGPU上で学習する
REQUIRE_GPU = True

# 78番ではDataParallelで複数GPUを使う想定
#
# True:
#   見えているGPUが2枚未満ならエラーにする
#
# False:
#   GPUが1枚でも実行する
#
# AIXで78番として実行するなら True のままでよい。
REQUIRE_MULTI_GPU = True

# 単語埋め込みをファインチューニングする
#
# 78番のもう1つの重要ポイント。
# True にすると、embedding.weight も学習で更新される。
FINE_TUNE_EMBEDDING = True

# 学習データを何件ずつまとめて処理するか
#
# DataParallelでは、このバッチが複数GPUに分割される。
# 例:
#   BATCH_SIZE=64, GPU 2枚
#   → 各GPUがおよそ32件ずつ処理する
BATCH_SIZE = 64

# 何周学習するか
NUM_EPOCHS = 5

# 学習率
#
# embeddingも更新するので、うまくいかなければ 1e-4 に下げてもよい。
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

        # CrossEntropyLossを使うので、ラベルはlong型にする
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


class BoWClassifier(nn.Module):
    """
    Bag of Words によるポジネガ分類モデル。

    1. input_idsを単語ベクトル列に変換する
    2. 単語ベクトルの平均を取って文ベクトルを作る
    3. 文ベクトルをlinear層に入れて2クラスのスコアを出す
    """

    def __init__(self, embedding_matrix, num_classes=2, fine_tune_embedding=True):
        super().__init__()

        # =========================
        # 78番の重要ポイント
        # =========================
        #
        # fine_tune_embedding=True:
        #   freeze=False
        #   → embedding.weight も学習で更新する
        #
        # fine_tune_embedding=False:
        #   freeze=True
        #   → embedding.weight は固定する
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=not fine_tune_embedding,
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

        # input_ids:
        #   [バッチサイズ, 文長]
        #
        # embeddings:
        #   [バッチサイズ, 文長, 300]
        embeddings = self.embedding(input_ids)

        # 文長方向に単語ベクトルを足す
        #
        # summed_vectors:
        #   [バッチサイズ, 300]
        summed_vectors = embeddings.sum(dim=1)

        # lengths:
        #   [バッチサイズ]
        #   ↓
        #   [バッチサイズ, 1]
        lengths = lengths.clamp(min=1).unsqueeze(1).to(summed_vectors.dtype)

        # 単語ベクトルの平均を取る
        #
        # sentence_vectors:
        #   [バッチサイズ, 300]
        sentence_vectors = summed_vectors / lengths

        # 2クラスのスコアを出す
        #
        # logits:
        #   [バッチサイズ, 2]
        logits = self.linear(sentence_vectors)

        return logits


def get_device():
    """
    GPUを使うdeviceを決める関数。

    CUDA_VISIBLE_DEVICESで見えているGPUの枚数も表示する。
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

        visible_gpu_count = torch.cuda.device_count()

        print(f"使用デバイス: {device}")
        print(f"PyTorchから見えているGPU数: {visible_gpu_count}")

        for gpu_id in range(visible_gpu_count):
            print(f"cuda:{gpu_id} = {torch.cuda.get_device_name(gpu_id)}")

        if REQUIRE_MULTI_GPU and visible_gpu_count < 2:
            raise RuntimeError(
                "DataParallelで複数GPUを使う設定ですが、"
                "PyTorchから見えているGPUが1枚以下です。\n"
                "例: CUDA_VISIBLE_DEVICES=1,2 uv run --no-sync python 78.py\n"
                "のように、空いているGPUをカンマ区切りで複数指定してください。"
            )

        return device

    if REQUIRE_GPU:
        raise RuntimeError(
            "GPUが使えません。AIXサーバなどGPUが使える環境で実行してください。"
        )

    device = torch.device("cpu")
    print(f"使用デバイス: {device}")
    return device


def unwrap_model(model):
    """
    DataParallelで包まれている場合は、中の元モデルを取り出す関数。

    DataParallelを使うと、
        model
    は DataParallel オブジェクトになり、元のBoWClassifierは
        model.module
    に入る。

    保存時や embedding.weight を直接確認するときに使う。
    """

    if isinstance(model, nn.DataParallel):
        return model.module

    return model


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
    print("78番: 単語埋め込みも更新し、DataParallelでマルチGPU学習します。")

    # =========================
    # deviceを決める
    # =========================

    device = get_device()

    visible_gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0

    # DataParallelで使うGPU番号。
    #
    # 注意:
    # CUDA_VISIBLE_DEVICES=1,2 として実行した場合でも、
    # Python内ではそれらは cuda:0, cuda:1 として見える。
    #
    # そのため、device_ids は [1, 2] ではなく [0, 1] にする。
    data_parallel_device_ids = list(range(visible_gpu_count))

    # =========================
    # 70番の単語埋め込みデータを読み込む
    # =========================

    # まずCPU上に読み込む。
    # その後 model.to(device) でGPUへ移す。
    data = torch.load(EMBEDDING_FILE, map_location="cpu")

    embedding_matrix = data["embedding_matrix"]
    token_to_id = data["token_to_id"]

    print(f"embedding_matrix の形: {embedding_matrix.shape}")
    print(f"語彙数: {len(token_to_id)}")
    print(f"単語埋め込みをファインチューニングするか: {FINE_TUNE_EMBEDDING}")

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

    model = BoWClassifier(
        embedding_matrix=embedding_matrix,
        fine_tune_embedding=FINE_TUNE_EMBEDDING,
    )

    # DataParallelを使う前に、元モデルをcuda:0へ移動する。
    #
    # DataParallelでは、元モデルのパラメータが device_ids[0]、
    # つまり cuda:0 にある必要がある。
    model = model.to(device)

    # =========================
    # DataParallelでモデルを包む
    # =========================

    if device.type == "cuda" and visible_gpu_count >= 2:
        print(f"DataParallelを使います。device_ids={data_parallel_device_ids}")

        model = nn.DataParallel(
            model,
            device_ids=data_parallel_device_ids,
            output_device=0,
        )
    else:
        print("DataParallelは使いません。")

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

    # DataParallelで包まれていても、中の元モデルを取り出して確認する
    base_model = unwrap_model(model)

    print(f"embedding.weight requires_grad: {base_model.embedding.weight.requires_grad}")
    print(f"linear.weight requires_grad: {base_model.linear.weight.requires_grad}")
    print(f"linear.bias requires_grad: {base_model.linear.bias.requires_grad}")

    # =========================
    # 損失関数と最適化手法を用意する
    # =========================

    criterion = nn.CrossEntropyLoss()

    # 78番では embedding.weight も requires_grad=True なので、
    # optimizerには embedding.weight, linear.weight, linear.bias が渡される。
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
            # batchはまずcuda:0へ送る。
            # DataParallelがそこから各GPUへ分割してくれる。
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()

            # DataParallelで包まれている場合、
            # ここで input_ids と lengths がbatch方向に分割され、
            # 各GPUへ送られて計算される。
            logits = model(input_ids, lengths)

            # logitsはoutput_device=0、つまりcuda:0に集約される。
            # labelsもcuda:0にあるので、そのままlossを計算できる。
            loss = criterion(logits, labels)

            # 78番では、
            #   embedding.weight
            #   linear.weight
            #   linear.bias
            # に対して勾配が計算される。
            #
            # DataParallel使用時は、各GPUで計算された勾配が集約される。
            loss.backward()

            # 集約された勾配を使って重みを更新する。
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

        # 1エポック終了後、devデータで評価する
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

    # DataParallelで包まれている場合、
    # model.state_dict() をそのまま保存するとキーに "module." が付く。
    #
    # 例:
    #   module.embedding.weight
    #   module.linear.weight
    #
    # あとで普通のBoWClassifierに読み込みやすくするため、
    # model.module.state_dict() を保存する。
    model_to_save = unwrap_model(model)

    # 保存するときはCPUに戻す。
    #
    # 78番ではembeddingも更新しているので、
    # model全体を保存する。
    #
    # 注意:
    #   embedding.weight が大きいので、
    #   bow_model_78.pt は大きくなる。
    #   chapter8/data/ はGitHubに上げないこと。
    model_state_dict = {
        key: value.cpu()
        for key, value in model_to_save.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict": model_state_dict,
            "num_classes": 2,
            "embedding_dim": embedding_matrix.size(1),
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "fine_tune_embedding": FINE_TUNE_EMBEDDING,
            "use_data_parallel": isinstance(model, nn.DataParallel),
            "visible_gpu_count": visible_gpu_count,
            "data_parallel_device_ids": data_parallel_device_ids,
            "device": str(device),
        },
        MODEL_OUTPUT_FILE,
    )

    print("学習済みモデルを保存しました")


if __name__ == "__main__":
    main()

#出力
# 78番: 単語埋め込みも更新し、DataParallelでマルチGPU学習します。
# 使用デバイス: cuda
# PyTorchから見えているGPU数: 2
# cuda:0 = NVIDIA RTX A6000
# cuda:1 = NVIDIA RTX A6000
# embedding_matrix の形: torch.Size([300001, 300])
# 語彙数: 300001
# 単語埋め込みをファインチューニングするか: True
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
# DataParallelを使います。device_ids=[0, 1]
# DataParallel(
#   (module): BoWClassifier(
#     (embedding): Embedding(300001, 300, padding_idx=0)
#     (linear): Linear(in_features=300, out_features=2, bias=True)
#   )
# )
# 全パラメータ数: 90000902
# 学習対象パラメータ数: 90000902
# embedding.weight requires_grad: True
# linear.weight requires_grad: True
# linear.bias requires_grad: True
# epoch 1 | step 100/1038 | train loss 0.6087 | train accuracy 0.6894
# epoch 1 | step 200/1038 | train loss 0.5382 | train accuracy 0.7606
# epoch 1 | step 300/1038 | train loss 0.4877 | train accuracy 0.7892
# epoch 1 | step 400/1038 | train loss 0.4479 | train accuracy 0.8120
# epoch 1 | step 500/1038 | train loss 0.4216 | train accuracy 0.8252
# epoch 1 | step 600/1038 | train loss 0.4043 | train accuracy 0.8339
# epoch 1 | step 700/1038 | train loss 0.3920 | train accuracy 0.8407
# epoch 1 | step 800/1038 | train loss 0.3820 | train accuracy 0.8469
# epoch 1 | step 900/1038 | train loss 0.3721 | train accuracy 0.8523
# epoch 1 | step 1000/1038 | train loss 0.3656 | train accuracy 0.8560
# [epoch 1 finished] train loss 0.3635 | train accuracy 0.8573 | dev loss 0.4163 | dev accuracy 0.8188 | time 50.24 sec
# epoch 2 | step 100/1038 | train loss 0.2406 | train accuracy 0.9120
# epoch 2 | step 200/1038 | train loss 0.2431 | train accuracy 0.9111
# epoch 2 | step 300/1038 | train loss 0.2424 | train accuracy 0.9100
# epoch 2 | step 400/1038 | train loss 0.2406 | train accuracy 0.9102
# epoch 2 | step 500/1038 | train loss 0.2396 | train accuracy 0.9109
# epoch 2 | step 600/1038 | train loss 0.2385 | train accuracy 0.9113
# epoch 2 | step 700/1038 | train loss 0.2374 | train accuracy 0.9123
# epoch 2 | step 800/1038 | train loss 0.2372 | train accuracy 0.9122
# epoch 2 | step 900/1038 | train loss 0.2367 | train accuracy 0.9128
# epoch 2 | step 1000/1038 | train loss 0.2367 | train accuracy 0.9125
# [epoch 2 finished] train loss 0.2380 | train accuracy 0.9121 | dev loss 0.4291 | dev accuracy 0.8096 | time 46.70 sec
# epoch 3 | step 100/1038 | train loss 0.1918 | train accuracy 0.9316
# epoch 3 | step 200/1038 | train loss 0.1947 | train accuracy 0.9291
# epoch 3 | step 300/1038 | train loss 0.1968 | train accuracy 0.9292
# epoch 3 | step 400/1038 | train loss 0.1980 | train accuracy 0.9285
# epoch 3 | step 500/1038 | train loss 0.1997 | train accuracy 0.9273
# epoch 3 | step 600/1038 | train loss 0.2002 | train accuracy 0.9265
# epoch 3 | step 700/1038 | train loss 0.1998 | train accuracy 0.9265
# epoch 3 | step 800/1038 | train loss 0.2013 | train accuracy 0.9261
# epoch 3 | step 900/1038 | train loss 0.2019 | train accuracy 0.9260
# epoch 3 | step 1000/1038 | train loss 0.2025 | train accuracy 0.9254
# [epoch 3 finished] train loss 0.2028 | train accuracy 0.9251 | dev loss 0.4622 | dev accuracy 0.8073 | time 46.66 sec
# epoch 4 | step 100/1038 | train loss 0.1571 | train accuracy 0.9427
# epoch 4 | step 200/1038 | train loss 0.1670 | train accuracy 0.9387
# epoch 4 | step 300/1038 | train loss 0.1690 | train accuracy 0.9367
# epoch 4 | step 400/1038 | train loss 0.1732 | train accuracy 0.9351
# epoch 4 | step 500/1038 | train loss 0.1779 | train accuracy 0.9330
# epoch 4 | step 600/1038 | train loss 0.1804 | train accuracy 0.9322
# epoch 4 | step 700/1038 | train loss 0.1809 | train accuracy 0.9320
# epoch 4 | step 800/1038 | train loss 0.1819 | train accuracy 0.9311
# epoch 4 | step 900/1038 | train loss 0.1828 | train accuracy 0.9309
# epoch 4 | step 1000/1038 | train loss 0.1844 | train accuracy 0.9301
# [epoch 4 finished] train loss 0.1853 | train accuracy 0.9296 | dev loss 0.4800 | dev accuracy 0.7970 | time 46.63 sec
# epoch 5 | step 100/1038 | train loss 0.1457 | train accuracy 0.9492
# epoch 5 | step 200/1038 | train loss 0.1544 | train accuracy 0.9423
# epoch 5 | step 300/1038 | train loss 0.1566 | train accuracy 0.9420
# epoch 5 | step 400/1038 | train loss 0.1589 | train accuracy 0.9405
# epoch 5 | step 500/1038 | train loss 0.1631 | train accuracy 0.9387
# epoch 5 | step 600/1038 | train loss 0.1658 | train accuracy 0.9368
# epoch 5 | step 700/1038 | train loss 0.1668 | train accuracy 0.9364
# epoch 5 | step 800/1038 | train loss 0.1697 | train accuracy 0.9351
# epoch 5 | step 900/1038 | train loss 0.1713 | train accuracy 0.9344
# epoch 5 | step 1000/1038 | train loss 0.1732 | train accuracy 0.9337
# [epoch 5 finished] train loss 0.1744 | train accuracy 0.9334 | dev loss 0.5206 | dev accuracy 0.7947 | time 46.62 sec
# 学習済みモデルを保存しました