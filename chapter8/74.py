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

# 73番で保存した学習済みモデル
# 例: nlp100/chapter8/data/bow_model_73.pt
MODEL_FILE = BASE_DIR / "data" / "bow_model_73.pt"

# 71番で使ったSST-2の訓練データ
# 例: nlp100/chapter7/SST-2/train.tsv
TRAIN_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "train.tsv"

# 71番で使ったSST-2の検証データ
# 例: nlp100/chapter7/SST-2/dev.tsv
DEV_FILE = PROJECT_ROOT / "chapter7" / "SST-2" / "dev.tsv"


# =========================
# 評価設定
# =========================

# 74番はCPUでOK。
# GPUを使いたい場合は True にする。
USE_GPU = False

# 評価時に何件ずつまとめて処理するか。
# 学習ではないので、ある程度大きめでもよい。
BATCH_SIZE = 64


def tokenize(text):
    """
    文をトークン列に変換する関数。

    71番・73番と同じ処理にする。
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

        # padding前の本来の長さ
        lengths.append(len(example["input_ids"]))

    # 長さの違うinput_idsを0でpaddingして、
    # [バッチサイズ, 最大文長] の形にする
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

    72番・73番と同じモデル構造にする。
    """

    def __init__(self, embedding_matrix, num_classes=2):
        super().__init__()

        # 70番で作った単語埋め込み行列を使う。
        #
        # freeze=True なので、評価時ももちろん更新されない。
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=True,
            padding_idx=0,
        )

        embedding_dim = embedding_matrix.size(1)

        # 73番で学習したlinear層の重みを、あとでここに読み込む。
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

        # 文ごとに単語ベクトルを足す
        summed_vectors = embeddings.sum(dim=1)

        # 各文の本来の長さで割るために、
        # [バッチサイズ] から [バッチサイズ, 1] に形を変える。
        lengths = lengths.clamp(min=1).unsqueeze(1).to(summed_vectors.dtype)

        # 単語ベクトルの平均を取る
        sentence_vectors = summed_vectors / lengths

        # 文ベクトルを2クラスのスコアに変換する
        logits = self.linear(sentence_vectors)

        return logits


def evaluate(model, data_loader, criterion, device):
    """
    指定されたデータに対して、lossとaccuracyを計算する関数。

    ここでは学習はしない。
    つまり、重み更新はしない。
    """

    # モデルを評価モードにする。
    #
    # 今回のモデルにはDropoutやBatchNormはないので大きな違いはないが、
    # 評価時には model.eval() と書くのが基本。
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # 評価時は勾配を計算しない。
    #
    # 学習しないので勾配は不要。
    # これを書くと、メモリ使用量を減らせて、少し高速になる。
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
    print("学習済みBoWモデルの正解率を計算します。")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    # =========================
    # 72番と同じモデル構造を作る
    # =========================

    model = BoWClassifier(embedding_matrix)

    # =========================
    # 73番で学習したlinear層の重みを読み込む
    # =========================

    checkpoint = torch.load(MODEL_FILE, map_location="cpu")

    model.linear.load_state_dict(checkpoint["linear_state_dict"])

    # モデルをCPUまたはGPUに移動する
    model = model.to(device)

    print(model)
    print("学習済みモデルを読み込みました")

    # =========================
    # loss関数を用意する
    # =========================

    # 評価時にもlossを計算したいので用意する。
    # accuracyだけ欲しいなら、loss計算は省略してもよい。
    criterion = nn.CrossEntropyLoss()

    # =========================
    # train/dev の正解率を計算する
    # =========================

    train_loss, train_accuracy = evaluate(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        device=device,
    )

    dev_loss, dev_accuracy = evaluate(
        model=model,
        data_loader=dev_loader,
        criterion=criterion,
        device=device,
    )

    print("\n評価結果")
    print(f"train loss: {train_loss:.4f}")
    print(f"train accuracy: {train_accuracy:.4f}")
    print(f"dev loss: {dev_loss:.4f}")
    print(f"dev accuracy: {dev_accuracy:.4f}")


if __name__ == "__main__":
    main()

# 出力
# 学習済みBoWモデルの正解率を計算します。
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
# BoWClassifier(
#   (embedding): Embedding(300001, 300, padding_idx=0)
#   (linear): Linear(in_features=300, out_features=2, bias=True)
# )
# 学習済みモデルを読み込みました

# 評価結果
# train loss: 0.3732
# train accuracy: 0.8411
# dev loss: 0.4578
# dev accuracy: 0.7924