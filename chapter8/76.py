from pathlib import Path
import csv

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

# 76番で学習したモデルの保存先
# 73番のモデルと区別するために bow_model_76.pt にする
MODEL_OUTPUT_FILE = BASE_DIR / "data" / "bow_model_76.pt"


# =========================
# 学習設定
# =========================

# 76番はCPUでOK。
# 77番でGPU上での学習が別途指定されているため。
USE_GPU = False

# 学習データを何件ずつまとめて処理するか
BATCH_SIZE = 64

# 何周学習するか
NUM_EPOCHS = 5

# 学習率。
LEARNING_RATE = 1e-3

# 何ステップごとに途中経過を表示するか
LOG_INTERVAL = 100


def tokenize(text):
    """
    文をトークン列に変換する関数。

    今回はSST-2がすでにある程度スペース区切りされているので、
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

        # CrossEntropyLossを使うので、ラベルはlong型の整数にする
        label = torch.tensor(example["label"], dtype=torch.long)

        return {
            "text": example["text"],
            "input_ids": input_ids,
            "label": label,
        }


def collate_batch(batch):
    """
    DataLoaderで複数事例を1つのミニバッチにまとめる関数。

    76番では、75番の仕様を使う。

    75番の仕様:
    1. トークン列の長さが長い順に事例を並び替える
    2. 一番長い系列長に合わせて、短い文を0でpaddingする
    3. input_ids, label, lengths をテンソルとして返す
    """

    # =========================
    # 1. 長さが長い順に並び替える
    # =========================
    #
    # batch はDatasetから取り出された複数事例のリスト。
    #
    # 例:
    # [
    #   {"text": ..., "input_ids": tensor([...]), "label": tensor(0)},
    #   {"text": ..., "input_ids": tensor([...]), "label": tensor(1)},
    #   ...
    # ]
    #
    # len(example["input_ids"]) が文のトークン数。
    # これが大きい順になるように並び替える。
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
    # 2. 並び替え後の順番で情報を集める
    # =========================

    for example in sorted_batch:
        texts.append(example["text"])
        input_ids_list.append(example["input_ids"])
        labels.append(example["label"])

        # padding前の本来の長さを保存する
        lengths.append(len(example["input_ids"]))

    # =========================
    # 3. 長さの違うinput_idsをpaddingする
    # =========================
    #
    # input_ids_list は長さがバラバラのテンソルのリスト。
    #
    # pad_sequenceを使うと、
    # 一番長い系列長にそろえて、足りない部分を0で埋める。
    #
    # batch_first=True により、出力形状は
    # [バッチサイズ, 最大文長]
    # になる。
    padded_input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0,
    )

    # labels は最初、
    #   [tensor(1), tensor(0), ...]
    # のようなリストなので、
    #   tensor([1, 0, ...])
    # にまとめる。
    labels = torch.stack(labels)

    # lengths は最初、
    #   [11, 7, 6, ...]
    # のようなPythonリストなので、
    #   tensor([11, 7, 6, ...])
    # に変換する。
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

        # 70番で作った単語埋め込み行列を使う。
        #
        # freeze=True にしているので、
        # 学習中に単語埋め込み行列は更新されない。
        #
        # つまり、問題文の
        # 「単語埋め込み行列のファインチューニングは行わない」
        # に対応している。
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=True,
            padding_idx=0,
        )

        # 単語ベクトルの次元数
        # GoogleNewsなら300
        embedding_dim = embedding_matrix.size(1)

        # 300次元の文ベクトルを2クラスのスコアに変換する層
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
        #
        # input_ids:
        #   [バッチサイズ, 文長]
        #
        # embeddings:
        #   [バッチサイズ, 文長, 300]
        embeddings = self.embedding(input_ids)

        # 文ごとに、単語ベクトルを足す
        #
        # embeddings.sum(dim=1) は、
        # 文長方向に足すという意味。
        #
        # summed_vectors:
        #   [バッチサイズ, 300]
        summed_vectors = embeddings.sum(dim=1)

        # 各文の長さで割れるように、
        # lengthsを [バッチサイズ] から [バッチサイズ, 1] に変える。
        #
        # 例:
        #   tensor([2, 3])
        #   ↓
        #   tensor([[2],
        #           [3]])
        #
        # clamp(min=1) は、念のため0で割るのを防ぐため。
        lengths = lengths.clamp(min=1).unsqueeze(1).to(summed_vectors.dtype)

        # 単語ベクトルの平均を取る
        #
        # sentence_vectors:
        #   [バッチサイズ, 300]
        sentence_vectors = summed_vectors / lengths

        # 文ベクトルを2クラスのスコアに変換する
        #
        # logits:
        #   [バッチサイズ, 2]
        logits = self.linear(sentence_vectors)

        return logits


def evaluate(model, data_loader, criterion, device):
    """
    指定されたデータに対して、lossとaccuracyを計算する関数。

    ここでは学習はしない。
    つまり、重み更新はしない。
    """

    # モデルを評価モードにする
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # 評価時は勾配を計算しない。
    # 学習しないので勾配は不要。
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            # モデルに入力して、2クラスのスコアを出す
            logits = model(input_ids, lengths)

            # lossを計算する
            loss = criterion(logits, labels)

            batch_size = labels.size(0)

            # lossはバッチ平均なので、
            # データ数をかけて合計lossに直してから足す。
            total_loss += loss.item() * batch_size

            # 各文について、スコアが大きい方を予測ラベルにする
            predicted_labels = torch.argmax(logits, dim=1)

            # 予測と正解が一致した数を足す
            total_correct += (predicted_labels == labels).sum().item()

            # 処理したデータ数を足す
            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def main():
    print("75番の仕様を入れてBoWモデルをミニバッチ学習します。")

    # =========================
    # deviceを決める
    # =========================

    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
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

    model = BoWClassifier(embedding_matrix)

    # モデルをCPUまたはGPUに移動する
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
        # モデルを学習モードにする
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for step, batch in enumerate(train_loader, start=1):
            # DataLoaderから取り出したテンソルをdeviceに移す
            input_ids = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            # 前回の勾配をリセットする
            optimizer.zero_grad()

            # モデルに入力して、2クラスのスコアを出す
            logits = model(input_ids, lengths)

            # 予測スコアと正解ラベルを比べてlossを計算する
            loss = criterion(logits, labels)

            # lossをもとに勾配を計算する
            loss.backward()

            # 勾配を使って重みを更新する
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

        # =========================
        # 1エポック終了時の訓練データでの結果
        # =========================

        train_loss = total_loss / total_examples
        train_accuracy = total_correct / total_examples

        # =========================
        # 1エポック終了後、devデータで評価する
        # =========================
        #
        # 76番の問題文では、学習したモデルの開発セットにおける正解率を求める。
        # そのため、各epoch終了後にdevのaccuracyも計算する。
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
            f"dev accuracy {dev_accuracy:.4f}"
        )

    # =========================
    # 学習済みモデルを保存する
    # =========================

    MODEL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 単語埋め込み行列は70番のembeddings.ptに保存済みで、
    # さらに学習中も固定している。
    #
    # そのため、ここでは学習されたlinear層だけ保存する。
    torch.save(
        {
            "linear_state_dict": model.linear.state_dict(),
            "num_classes": 2,
            "embedding_dim": embedding_matrix.size(1),
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        MODEL_OUTPUT_FILE,
    )

    print("学習済みモデルを保存しました")


if __name__ == "__main__":
    main()

# 出力
# 75番の仕様を入れてBoWモデルをミニバッチ学習します。
# 使用デバイス: cpu
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
# epoch 1 | step 100/1038 | train loss 0.6348 | train accuracy 0.6480
# epoch 1 | step 200/1038 | train loss 0.5975 | train accuracy 0.7126
# epoch 1 | step 300/1038 | train loss 0.5681 | train accuracy 0.7433
# epoch 1 | step 400/1038 | train loss 0.5486 | train accuracy 0.7579
# epoch 1 | step 500/1038 | train loss 0.5319 | train accuracy 0.7701
# epoch 1 | step 600/1038 | train loss 0.5193 | train accuracy 0.7778
# epoch 1 | step 700/1038 | train loss 0.5085 | train accuracy 0.7835
# epoch 1 | step 800/1038 | train loss 0.4988 | train accuracy 0.7899
# epoch 1 | step 900/1038 | train loss 0.4907 | train accuracy 0.7931
# epoch 1 | step 1000/1038 | train loss 0.4839 | train accuracy 0.7962
# [epoch 1 finished] train loss 0.4816 | train accuracy 0.7969 | dev loss 0.5032 | dev accuracy 0.7878
# epoch 2 | step 100/1038 | train loss 0.4104 | train accuracy 0.8313
# epoch 2 | step 200/1038 | train loss 0.4118 | train accuracy 0.8282
# epoch 2 | step 300/1038 | train loss 0.4098 | train accuracy 0.8286
# epoch 2 | step 400/1038 | train loss 0.4088 | train accuracy 0.8284
# epoch 2 | step 500/1038 | train loss 0.4054 | train accuracy 0.8302
# epoch 2 | step 600/1038 | train loss 0.4047 | train accuracy 0.8308
# epoch 2 | step 700/1038 | train loss 0.4037 | train accuracy 0.8304
# epoch 2 | step 800/1038 | train loss 0.4022 | train accuracy 0.8312
# epoch 2 | step 900/1038 | train loss 0.4012 | train accuracy 0.8314
# epoch 2 | step 1000/1038 | train loss 0.4004 | train accuracy 0.8320
# [epoch 2 finished] train loss 0.4000 | train accuracy 0.8319 | dev loss 0.4774 | dev accuracy 0.7821
# epoch 3 | step 100/1038 | train loss 0.3870 | train accuracy 0.8320
# epoch 3 | step 200/1038 | train loss 0.3918 | train accuracy 0.8322
# epoch 3 | step 300/1038 | train loss 0.3875 | train accuracy 0.8346
# epoch 3 | step 400/1038 | train loss 0.3855 | train accuracy 0.8361
# epoch 3 | step 500/1038 | train loss 0.3866 | train accuracy 0.8367
# epoch 3 | step 600/1038 | train loss 0.3862 | train accuracy 0.8363
# epoch 3 | step 700/1038 | train loss 0.3862 | train accuracy 0.8358
# epoch 3 | step 800/1038 | train loss 0.3851 | train accuracy 0.8366
# epoch 3 | step 900/1038 | train loss 0.3848 | train accuracy 0.8366
# epoch 3 | step 1000/1038 | train loss 0.3846 | train accuracy 0.8368
# [epoch 3 finished] train loss 0.3848 | train accuracy 0.8367 | dev loss 0.4679 | dev accuracy 0.7844
# epoch 4 | step 100/1038 | train loss 0.3753 | train accuracy 0.8400
# epoch 4 | step 200/1038 | train loss 0.3794 | train accuracy 0.8407
# epoch 4 | step 300/1038 | train loss 0.3728 | train accuracy 0.8421
# epoch 4 | step 400/1038 | train loss 0.3762 | train accuracy 0.8390
# epoch 4 | step 500/1038 | train loss 0.3752 | train accuracy 0.8402
# epoch 4 | step 600/1038 | train loss 0.3760 | train accuracy 0.8404
# epoch 4 | step 700/1038 | train loss 0.3757 | train accuracy 0.8410
# epoch 4 | step 800/1038 | train loss 0.3764 | train accuracy 0.8402
# epoch 4 | step 900/1038 | train loss 0.3768 | train accuracy 0.8395
# epoch 4 | step 1000/1038 | train loss 0.3782 | train accuracy 0.8395
# [epoch 4 finished] train loss 0.3785 | train accuracy 0.8397 | dev loss 0.4593 | dev accuracy 0.7924
# epoch 5 | step 100/1038 | train loss 0.3590 | train accuracy 0.8494
# epoch 5 | step 200/1038 | train loss 0.3688 | train accuracy 0.8445
# epoch 5 | step 300/1038 | train loss 0.3713 | train accuracy 0.8421
# epoch 5 | step 400/1038 | train loss 0.3694 | train accuracy 0.8422
# epoch 5 | step 500/1038 | train loss 0.3720 | train accuracy 0.8403
# epoch 5 | step 600/1038 | train loss 0.3733 | train accuracy 0.8403
# epoch 5 | step 700/1038 | train loss 0.3723 | train accuracy 0.8410
# epoch 5 | step 800/1038 | train loss 0.3736 | train accuracy 0.8410
# epoch 5 | step 900/1038 | train loss 0.3745 | train accuracy 0.8409
# epoch 5 | step 1000/1038 | train loss 0.3748 | train accuracy 0.8411
# [epoch 5 finished] train loss 0.3752 | train accuracy 0.8409 | dev loss 0.4572 | dev accuracy 0.7959
# 学習済みモデルを保存しました
