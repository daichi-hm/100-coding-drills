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
# 例: nlp100
PROJECT_ROOT = BASE_DIR.parent

# 70番で作成した単語埋め込みデータ
# 例: nlp100/chapter8/data/embeddings.pt
EMBEDDING_FILE = BASE_DIR / "data" / "embeddings.pt"

# 71番で使ったSST-2のtrain.tsv
# 例: nlp100/chapter7/SST-2/train.tsv
TRAIN_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "train.tsv"

# 73番で学習したモデルの保存先
# 今回は単語埋め込み行列は保存せず、学習したlinear層だけ保存する
MODEL_OUTPUT_FILE = BASE_DIR / "data" / "bow_model_73.pt"


# =========================
# 学習設定
# =========================

# 73番はCPUでOK。
# GPUを使いたい場合は True にする。
USE_GPU = False

# 学習データを何件ずつまとめて処理するか
BATCH_SIZE = 64

# 何周学習するか
NUM_EPOCHS = 5

# 学習率。
# 大きすぎると学習が不安定になり、小さすぎると学習が進みにくい。
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

    文の長さがバラバラなので、0番の<PAD>で長さをそろえる。
    """

    texts = []
    input_ids_list = []
    labels = []
    lengths = []

    for example in batch:
        texts.append(example["text"])
        input_ids_list.append(example["input_ids"])
        labels.append(example["label"])

        # padding前の本来の長さを保存する
        lengths.append(len(example["input_ids"]))

    # 長さの違うinput_idsを0でpaddingして、
    # [バッチサイズ, 最大文長] の形にする
    padded_input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=0,
    )

    # ラベルを [バッチサイズ] のテンソルにまとめる
    labels = torch.stack(labels)

    # 各文の長さを [バッチサイズ] のテンソルにする
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

        # 70番で作った単語埋め込み行列を使う
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


def main():
    print("BoWモデルを訓練セットで学習します。")

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
    # 71番と同じように訓練データを読み込む
    # =========================

    train_examples = load_sst_file(TRAIN_FILE, token_to_id)

    train_dataset = SSTDataset(train_examples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )

    print(f"train_dataset のデータ数: {len(train_dataset)}")
    print(f"バッチサイズ: {BATCH_SIZE}")
    print(f"1エポックあたりのバッチ数: {len(train_loader)}")

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

    # CrossEntropyLossは、分類問題でよく使う損失関数。
    #
    # logits:
    #   [バッチサイズ, クラス数]
    #
    # labels:
    #   [バッチサイズ]
    #
    # の形を受け取る。
    criterion = nn.CrossEntropyLoss()

    # optimizerは、重みを更新するための道具。
    #
    # embeddingはfreeze=Trueなのでrequires_grad=False。
    # そのため、ここではrequires_grad=Trueのパラメータだけを渡す。
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
                    f"loss {average_loss:.4f} | "
                    f"accuracy {accuracy:.4f}"
                )

        # =========================
        # 1エポック終了時の結果を表示する
        # =========================

        epoch_loss = total_loss / total_examples
        epoch_accuracy = total_correct / total_examples

        print(
            f"[epoch {epoch} finished] "
            f"loss {epoch_loss:.4f} | "
            f"accuracy {epoch_accuracy:.4f}"
        )

    # =========================
    # 学習済みモデルを保存する
    # =========================

    MODEL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 単語埋め込み行列は70番のembeddings.ptに保存済みで、
    # さらに学習中も固定している。
    #
    # そのため、ここでは学習されたlinear層だけ保存する。
    # これなら保存ファイルが小さくなる。
    torch.save(
        {
            "linear_state_dict": model.linear.state_dict(),
            "num_classes": 2,
            "embedding_dim": embedding_matrix.size(1),
        },
        MODEL_OUTPUT_FILE,
    )

    print("学習済みモデルを保存しました")


if __name__ == "__main__":
    main()

# 出力
# BoWモデルを訓練セットで学習します。
# 使用デバイス: cpu
# embedding_matrix の形: torch.Size([300001, 300])
# 語彙数: 300001
# train.tsv: 元データ数 = 67349
# train.tsv: 削除したデータ数 = 953
# train.tsv: 使用するデータ数 = 66396
# train_dataset のデータ数: 66396
# バッチサイズ: 64
# 1エポックあたりのバッチ数: 1038
# BoWClassifier(
#   (embedding): Embedding(300001, 300, padding_idx=0)
#   (linear): Linear(in_features=300, out_features=2, bias=True)
# )
# 全パラメータ数: 90000902
# 学習対象パラメータ数: 602
# epoch 1 | step 100/1038 | loss 0.6327 | accuracy 0.6748
# epoch 1 | step 200/1038 | loss 0.5928 | accuracy 0.7338
# epoch 1 | step 300/1038 | loss 0.5654 | accuracy 0.7568
# epoch 1 | step 400/1038 | loss 0.5447 | accuracy 0.7737
# epoch 1 | step 500/1038 | loss 0.5280 | accuracy 0.7824
# epoch 1 | step 600/1038 | loss 0.5149 | accuracy 0.7878
# epoch 1 | step 700/1038 | loss 0.5047 | accuracy 0.7933
# epoch 1 | step 800/1038 | loss 0.4969 | accuracy 0.7957
# epoch 1 | step 900/1038 | loss 0.4893 | accuracy 0.7988
# epoch 1 | step 1000/1038 | loss 0.4824 | accuracy 0.8020
# [epoch 1 finished] loss 0.4801 | accuracy 0.8030
# epoch 2 | step 100/1038 | loss 0.4175 | accuracy 0.8333
# epoch 2 | step 200/1038 | loss 0.4164 | accuracy 0.8262
# epoch 2 | step 300/1038 | loss 0.4127 | accuracy 0.8294
# epoch 2 | step 400/1038 | loss 0.4104 | accuracy 0.8293
# epoch 2 | step 500/1038 | loss 0.4079 | accuracy 0.8304
# epoch 2 | step 600/1038 | loss 0.4051 | accuracy 0.8305
# epoch 2 | step 700/1038 | loss 0.4040 | accuracy 0.8314
# epoch 2 | step 800/1038 | loss 0.4028 | accuracy 0.8308
# epoch 2 | step 900/1038 | loss 0.4007 | accuracy 0.8314
# epoch 2 | step 1000/1038 | loss 0.4008 | accuracy 0.8313
# [epoch 2 finished] loss 0.4000 | accuracy 0.8316
# epoch 3 | step 100/1038 | loss 0.3923 | accuracy 0.8303
# epoch 3 | step 200/1038 | loss 0.3942 | accuracy 0.8298
# epoch 3 | step 300/1038 | loss 0.3877 | accuracy 0.8354
# epoch 3 | step 400/1038 | loss 0.3898 | accuracy 0.8362
# epoch 3 | step 500/1038 | loss 0.3888 | accuracy 0.8367
# epoch 3 | step 600/1038 | loss 0.3890 | accuracy 0.8357
# epoch 3 | step 700/1038 | loss 0.3888 | accuracy 0.8357
# epoch 3 | step 800/1038 | loss 0.3880 | accuracy 0.8355
# epoch 3 | step 900/1038 | loss 0.3862 | accuracy 0.8366
# epoch 3 | step 1000/1038 | loss 0.3851 | accuracy 0.8369
# [epoch 3 finished] loss 0.3849 | accuracy 0.8370
# epoch 4 | step 100/1038 | loss 0.3768 | accuracy 0.8383
# epoch 4 | step 200/1038 | loss 0.3770 | accuracy 0.8413
# epoch 4 | step 300/1038 | loss 0.3732 | accuracy 0.8413
# epoch 4 | step 400/1038 | loss 0.3749 | accuracy 0.8409
# epoch 4 | step 500/1038 | loss 0.3761 | accuracy 0.8405
# epoch 4 | step 600/1038 | loss 0.3771 | accuracy 0.8398
# epoch 4 | step 700/1038 | loss 0.3776 | accuracy 0.8407
# epoch 4 | step 800/1038 | loss 0.3777 | accuracy 0.8405
# epoch 4 | step 900/1038 | loss 0.3783 | accuracy 0.8399
# epoch 4 | step 1000/1038 | loss 0.3786 | accuracy 0.8395
# [epoch 4 finished] loss 0.3785 | accuracy 0.8398
# epoch 5 | step 100/1038 | loss 0.3933 | accuracy 0.8378
# epoch 5 | step 200/1038 | loss 0.3887 | accuracy 0.8359
# epoch 5 | step 300/1038 | loss 0.3839 | accuracy 0.8389
# epoch 5 | step 400/1038 | loss 0.3793 | accuracy 0.8409
# epoch 5 | step 500/1038 | loss 0.3771 | accuracy 0.8415
# epoch 5 | step 600/1038 | loss 0.3757 | accuracy 0.8412
# epoch 5 | step 700/1038 | loss 0.3751 | accuracy 0.8413
# epoch 5 | step 800/1038 | loss 0.3752 | accuracy 0.8412
# epoch 5 | step 900/1038 | loss 0.3754 | accuracy 0.8412
# epoch 5 | step 1000/1038 | loss 0.3757 | accuracy 0.8410
# [epoch 5 finished] loss 0.3752 | accuracy 0.8412
# 学習済みモデルを保存しました
