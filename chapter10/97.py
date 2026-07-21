from pathlib import Path
import csv
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer


# ============================================================
# 設定
# ============================================================

MODEL_NAME = "openai-community/gpt2-medium"

# 1文を最大何トークンまで使用するか
# SST-2は比較的短い文が多いため、今回は128とする
MAX_LENGTH = 128

# GPT-2で文をベクトルに変換するときのバッチサイズ
ENCODE_BATCH_SIZE = 16

# 作成済みの文ベクトルを分類器に入力するときのバッチサイズ
# 分類器は小さいので、GPT-2より大きなバッチサイズを使える
CLASSIFIER_BATCH_SIZE = 256

# 分類器を学習するエポック数
EPOCHS = 10

# 分類器の学習率
LEARNING_RATE = 1e-3

# 乱数を固定するための値
SEED = 42

# Trueにすると、一部のデータだけを使って動作確認する
# 最終的な実験ではFalseにする
DEBUG = False


# ============================================================
# 乱数の固定
# ============================================================

def set_seed(seed: int) -> None:
    """
    実行するたびに結果が大きく変わらないように、
    PythonとPyTorchの乱数を固定する。
    """

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SST-2の読み込み
# ============================================================

def load_sst2(path: Path) -> tuple[list[str], torch.Tensor]:
    """
    SST-2のTSVファイルを読み込む。

    SST-2のファイルは、次のような形式になっている。

    sentence    label
    a stirring ...    1
    unflinchingly bleak ...    0

    戻り値:
        texts:
            文を格納したリスト

        labels:
            正解ラベルを格納したTensor
            0 = negative
            1 = positive
    """

    texts: list[str] = []
    labels: list[int] = []

    # TSVなので、delimiter="\t"を指定する
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            texts.append(row["sentence"])
            labels.append(int(row["label"]))

    # CrossEntropyLossで使うラベルはtorch.longにする
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return texts, labels_tensor


# ============================================================
# GPT-2による文ベクトルの作成
# ============================================================

def encode_texts(
    texts: list[str],
    tokenizer: GPT2Tokenizer,
    encoder: GPT2Model,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    複数の文をGPT-2に入力し、
    各文を1本のベクトルに変換する。

    戻り値のshape:
        [文の数, GPT-2のhidden size]
    """

    # 各バッチで作った文ベクトルを保存するリスト
    all_embeddings: list[torch.Tensor] = []

    # GPT-2を推論モードにする
    # Dropoutなど、学習時だけ使われる処理を停止する
    encoder.eval()

    # batch_size個ずつ文を処理する
    #tqdmは進捗バーを表示するためのもの
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding",
    ):
        batch_texts = texts[start : start + batch_size]

        # ----------------------------------------------------
        # 文をトークンIDに変換する
        # ----------------------------------------------------
        tokens = tokenizer(
            batch_texts,

            # バッチ内で長さをそろえる
            padding=True,

            # MAX_LENGTHを超える部分は切り捨てる
            truncation=True,

            max_length=MAX_LENGTH,

            # PyTorchのTensorとして返す
            return_tensors="pt",
        )

        # GPUまたはCPUに送る
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # ----------------------------------------------------
        # GPT-2で文をベクトル化する
        # ----------------------------------------------------

        # GPT-2本体は学習しないので、勾配を記録しない
        # これによりGPUメモリの使用量も減る
        with torch.no_grad():
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,

                # 過去のKey・Valueのキャッシュは生成タスク用なので、
                # 今回は使用しない
                use_cache=False,
            )

            # shape:
            # [バッチサイズ, トークン数, hidden size]
            hidden_states = outputs.last_hidden_state

            # ------------------------------------------------
            # 各文の「最後の実トークン」の位置を求める
            # ------------------------------------------------
            #
            # attention_maskの例:
            #
            # 文A: [1, 1, 1, 1, 0, 0]
            # 文B: [1, 1, 1, 1, 1, 1]
            #
            # 1の数を合計し、1を引くと、
            # 最後の実トークンの位置になる。
            #
            # 文A: 4 - 1 = 3
            # 文B: 6 - 1 = 5

            last_token_indices = attention_mask.sum(dim=1) - 1

            # バッチ内の文番号
            # 例: batch_size=3なら [0, 1, 2]
            batch_indices = torch.arange(
                hidden_states.size(0),
                device=device,
            )

            # 各文について、最後の実トークンのベクトルを取り出す
            #
            # 結果のshape:
            # [バッチサイズ, hidden size]
            sentence_embeddings = hidden_states[
                batch_indices,
                last_token_indices,
            ]

        # GPUメモリに置き続けないよう、CPUへ戻す
        all_embeddings.append(sentence_embeddings.cpu())

    # 各バッチの結果を縦方向に結合する
    #
    # 最終的なshape:
    # [全文数, hidden size]
    return torch.cat(all_embeddings, dim=0)


# ============================================================
# 感情分類器
# ============================================================

class SentimentClassifier(nn.Module):
    """
    GPT-2が作った文ベクトルを受け取り、
    negative / positiveの2クラスを予測する分類器。

    学習するのは、このLinear層だけ。
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()

        # 入力:
        #   GPT-2の文ベクトル
        #
        # 出力:
        #   2クラス分のスコア
        #
        # 出力0番目 = negativeのスコア
        # 出力1番目 = positiveのスコア
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddingsのshape:
            [バッチサイズ, hidden size]

        戻り値logitsのshape:
            [バッチサイズ, 2]
        """

        return self.linear(embeddings)


# ============================================================
# 開発データでの正解率の計算
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    開発データに対する正解率を求める。
    """

    # 評価モードに切り替える
    model.eval()

    correct = 0
    total = 0

    for embeddings, labels in data_loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        # 各クラスのスコアを計算
        logits = model(embeddings)

        # スコアが最大のクラスを予測結果とする
        #
        # logitsの例:
        # [[-0.3, 1.2],
        #  [ 0.8, 0.1]]
        #
        # predictions:
        # [1, 0]
        predictions = logits.argmax(dim=1)

        # 正解した個数を数える
        correct += (predictions == labels).sum().item()

        # 全データ数を数える
        total += labels.size(0)

    return correct / total


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    set_seed(SEED)

    # --------------------------------------------------------
    # SST-2のパスを作る
    # --------------------------------------------------------
    #
    chapter10_dir = Path(__file__).resolve().parent

    # chapter10の1つ上に移動し、chapter7/SST-2へ進む
    sst2_dir = (
        chapter10_dir.parent
        / "chapter7"
        / "SST-2"
    )

    train_path = sst2_dir / "train.tsv"
    dev_path = sst2_dir / "dev.tsv"

    # ファイルが存在しない場合は、
    # 分かりやすいエラーメッセージを表示する
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            "SST-2が見つかりません。"
            "次の配置を確認してください。\n"
            f"train: {train_path}\n"
            f"dev:   {dev_path}"
        )

    # --------------------------------------------------------
    # SST-2を読み込む
    # --------------------------------------------------------

    train_texts, train_labels = load_sst2(train_path)
    dev_texts, dev_labels = load_sst2(dev_path)

    # DEBUG=Trueの場合は、一部のデータだけを使う
    if DEBUG:
        train_texts = train_texts[:1000]
        train_labels = train_labels[:1000]

        dev_texts = dev_texts[:200]
        dev_labels = dev_labels[:200]

    print(f"train size: {len(train_texts)}")
    print(f"dev size:   {len(dev_texts)}")

    # --------------------------------------------------------
    # GPUが使える場合はGPU、なければCPUを使う
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"device: {device}")

    # --------------------------------------------------------
    # GPT-2のTokenizerを読み込む
    # --------------------------------------------------------

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # GPT-2には元々pad tokenがない
    #
    # 複数の文をバッチにして長さをそろえるため、
    # eos tokenをpad tokenとしても利用する
    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # GPT-2本体を読み込む
    # --------------------------------------------------------

    # GPT2LMHeadModelではなくGPT2Modelを使う
    #
    # 今回必要なのは「次の単語の予測結果」ではなく、
    # GPT-2内部のhidden stateだから
    encoder = GPT2Model.from_pretrained(MODEL_NAME)

    # paddingに使うトークンIDをモデルにも設定する
    encoder.config.pad_token_id = tokenizer.pad_token_id

    encoder.to(device)

    # --------------------------------------------------------
    # GPT-2本体を固定する
    # --------------------------------------------------------

    # requires_grad=Falseにすることで、
    # GPT-2のパラメータを学習対象から外す
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    # --------------------------------------------------------
    # SST-2の全ての文をGPT-2のベクトルに変換する
    # --------------------------------------------------------

    print("trainデータをGPT-2のベクトルに変換します。")

    train_embeddings = encode_texts(
        train_texts,
        tokenizer,
        encoder,
        device,
        ENCODE_BATCH_SIZE,
    )

    print("devデータをGPT-2のベクトルに変換します。")

    dev_embeddings = encode_texts(
        dev_texts,
        tokenizer,
        encoder,
        device,
        ENCODE_BATCH_SIZE,
    )

    print(f"train embeddings: {train_embeddings.shape}")
    print(f"dev embeddings:   {dev_embeddings.shape}")

    # --------------------------------------------------------
    # これ以降はGPT-2を使わないので削除する
    # --------------------------------------------------------

    del encoder

    # 未使用になったGPUメモリを解放する
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 文ベクトルとラベルをDatasetにまとめる
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        train_embeddings,
        train_labels,
    )

    dev_dataset = TensorDataset(
        dev_embeddings,
        dev_labels,
    )

    # --------------------------------------------------------
    # DataLoaderを作成する
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=CLASSIFIER_BATCH_SIZE,

        # 学習データはエポックごとに順番を変える
        shuffle=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=CLASSIFIER_BATCH_SIZE,

        # 評価データは順番を変える必要がない
        shuffle=False,
    )

    # --------------------------------------------------------
    # 分類器を作る
    # --------------------------------------------------------

    # train_embeddingsのshapeは
    # [データ数, hidden size]
    #
    # size(1)でhidden sizeを取得するため、
    # 1024などの数値を直接書かなくてもよい
    classifier = SentimentClassifier(
        hidden_size=train_embeddings.size(1)
    ).to(device)

    # --------------------------------------------------------
    # 損失関数とOptimizer
    # --------------------------------------------------------

    # 2クラス分類なのでCrossEntropyLossを使う
    loss_fn = nn.CrossEntropyLoss()

    # 学習対象はclassifier.parameters()だけ
    #
    # つまり、GPT-2ではなくLinear層だけが更新される
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=LEARNING_RATE,
    )

    best_dev_accuracy = 0.0

    # 最も正解率が高かった分類器を保存するパス
    model_path = chapter10_dir / "97_classifier.pt"

    # ========================================================
    # 分類器の学習
    # ========================================================

    for epoch in range(1, EPOCHS + 1):
        # 学習モードに切り替える
        classifier.train()

        total_loss = 0.0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # 前回のバッチで計算された勾配をリセット
            optimizer.zero_grad()

            # 文ベクトルから2クラスのスコアを計算
            logits = classifier(embeddings)

            # 予測と正解ラベルの誤差を計算
            loss = loss_fn(logits, labels)

            # 誤差から勾配を計算
            loss.backward()

            # Linear層のパラメータを更新
            optimizer.step()

            # バッチ平均のlossなので、
            # バッチ内のデータ数を掛けて合計する
            total_loss += loss.item() * labels.size(0)

        # 1データあたりの平均loss
        average_loss = total_loss / len(train_dataset)

        # 開発データの正解率
        dev_accuracy = evaluate(
            classifier,
            dev_loader,
            device,
        )

        print(
            f"epoch {epoch:2d} | "
            f"train loss: {average_loss:.4f} | "
            f"dev accuracy: {dev_accuracy:.4f}"
        )

        # これまでで最も良い正解率なら保存する
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy

            torch.save(
                classifier.state_dict(),
                model_path,
            )

    print(f"best dev accuracy: {best_dev_accuracy:.4f}")


# このファイルが直接実行された場合だけmain()を呼ぶ
if __name__ == "__main__":
    main()

#出力
# train size: 67349
# dev size:   872
# device: cuda
# trainデータをGPT-2のベクトルに変換します。
# Encoding: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 4210/4210 [03:19<00:00, 21.06it/s]
# devデータをGPT-2のベクトルに変換します。
# Encoding: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 55/55 [00:03<00:00, 16.98it/s]
# train embeddings: torch.Size([67349, 1024])
# dev embeddings:   torch.Size([872, 1024])
# epoch  1 | train loss: 0.5042 | dev accuracy: 0.8211
# epoch  2 | train loss: 0.3906 | dev accuracy: 0.8475
# epoch  3 | train loss: 0.3638 | dev accuracy: 0.8509
# epoch  4 | train loss: 0.3509 | dev accuracy: 0.8521
# epoch  5 | train loss: 0.3401 | dev accuracy: 0.8601
# epoch  6 | train loss: 0.3342 | dev accuracy: 0.8624
# epoch  7 | train loss: 0.3308 | dev accuracy: 0.8658
# epoch  8 | train loss: 0.3258 | dev accuracy: 0.8601
# epoch  9 | train loss: 0.3213 | dev accuracy: 0.8544
# epoch 10 | train loss: 0.3210 | dev accuracy: 0.8635
# best dev accuracy: 0.8658