from pathlib import Path
import csv
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ============================================================
# 設定
# ============================================================

# 授業指定のモデル
MODEL_NAME = "openai-community/gpt2-medium"

# プロンプト + 正解ラベルの最大トークン数
MAX_LENGTH = 128

# 1回の学習で8件をまとめてGPUに入力する
TRAIN_BATCH_SIZE = 8

# 評価時は逆伝播をしないため、学習時より大きくできることが多い
EVAL_BATCH_SIZE = 16

EPOCHS = 1
LEARNING_RATE = 5e-5
SEED = 42

# 最初の動作確認ではTrueにする
DEBUG = False


# ============================================================
# 乱数の固定
# ============================================================

def set_seed(seed: int) -> None:
    """実行ごとの結果のばらつきを小さくする。"""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SST-2の読み込み
# ============================================================

def load_sst2(path: Path) -> tuple[list[str], list[int]]:
    """SST-2のTSVファイルから文とラベルを読み込む。"""

    texts: list[str] = []
    labels: list[int] = []

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            texts.append(row["sentence"])
            labels.append(int(row["label"]))

    return texts, labels


# ============================================================
# 96番と同じ形式のプロンプト
# ============================================================

def build_prompt(sentence: str) -> str:
    """分類対象の文を含む感情分析用プロンプトを作る。"""

    return (
        "Classify the sentiment of the movie review as positive or negative.\n\n"
        "Review: I loved this movie.\n"
        "Sentiment: positive\n\n"
        "Review: This movie was boring and terrible.\n"
        "Sentiment: negative\n\n"
        f"Review: {sentence}\n"
        "Sentiment:"
    )


def label_to_answer(label: int) -> str:
    """SST-2の0/1ラベルを生成させる文字列へ変換する。"""

    if label == 1:
        return " positive"

    return " negative"


# ============================================================
# 学習用Dataset
# ============================================================

class SentimentFineTuningDataset(Dataset):
    """SST-2の1件をGPT-2の言語モデル学習形式へ変換する。"""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: GPT2Tokenizer,
        max_length: int,
    ) -> None:
        super().__init__()

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """データ数を返す。"""

        return len(self.texts)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        """index番目のデータを学習用Tensorへ変換する。"""

        prompt = build_prompt(self.texts[index])
        answer = label_to_answer(self.labels[index])

        # プロンプトをトークンIDへ変換
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        # 正解のpositive / negativeをトークンIDへ変換
        answer_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
        )

        # 正解の後ろに文章終了トークンを付ける
        answer_ids.append(self.tokenizer.eos_token_id)

        # 正解部分を残した上で、プロンプトに使える長さを求める
        max_prompt_length = self.max_length - len(answer_ids)

        if max_prompt_length <= 0:
            raise ValueError("MAX_LENGTHが短すぎます。")

        # 長すぎる場合は末尾側を残す
        if len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]

        # GPT-2にはプロンプトと正解を両方入力する
        input_ids = prompt_ids + answer_ids

        # この時点ではpadding前なので全位置が実トークン
        attention_mask = [1] * len(input_ids)

        # プロンプト部分はlossの対象外にする
        # positive / negativeとEOSだけを正解として採点する
        lm_labels = (
            [-100] * len(prompt_ids)
            + answer_ids
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                lm_labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# バッチ作成時のpadding
# ============================================================

class CausalLMCollator:
    """長さの異なるデータをpaddingして1つのバッチへまとめる。"""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        batch: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """DataLoaderがバッチを作るときに呼ばれる。"""

        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )

        # padding位置もloss計算から除外する
        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================
# 開発データのloss計算
# ============================================================

@torch.no_grad()
def calculate_loss(
    model: GPT2LMHeadModel,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    """開発データに対する平均lossを計算する。"""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in tqdm(
        data_loader,
        desc="Validation loss",
    ):
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # mixed precisionは速度向上のため残す
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(**batch)

        batch_size = batch["input_ids"].size(0)

        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


# ============================================================
# 生成結果から感情ラベルを取得
# ============================================================

def extract_sentiment(generated_text: str) -> str:
    """生成文字列の先頭からpositive / negativeを判定する。"""

    normalized = generated_text.strip().lower()

    if normalized.startswith("positive"):
        return "positive"

    if normalized.startswith("negative"):
        return "negative"

    return "unknown"


# ============================================================
# 開発データの生成精度
# ============================================================

@torch.no_grad()
def evaluate_accuracy(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
) -> float:
    """positive / negativeを実際に生成させて正解率を求める。"""

    model.eval()

    correct = 0

    for sentence, label in tqdm(
        zip(texts, labels),
        total=len(texts),
        desc="Generation accuracy",
    ):
        prompt = build_prompt(sentence)

        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        if len(prompt_ids) > MAX_LENGTH:
            prompt_ids = prompt_ids[-MAX_LENGTH:]

        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=device,
        )

        attention_mask = torch.ones_like(input_ids)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_length = input_ids.size(1)

        answer_ids = generated_ids[
            0,
            prompt_length:,
        ]

        generated_text = tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
        )

        prediction = extract_sentiment(
            generated_text
        )

        gold = (
            "positive"
            if label == 1
            else "negative"
        )

        if prediction == gold:
            correct += 1

    return correct / len(texts)


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    set_seed(SEED)

    # --------------------------------------------------------
    # ファイルの場所
    # --------------------------------------------------------

    chapter10_dir = Path(__file__).resolve().parent

    sst2_dir = (
        chapter10_dir.parent
        / "chapter7"
        / "SST-2"
    )

    train_path = sst2_dir / "train.tsv"
    dev_path = sst2_dir / "dev.tsv"

    save_dir = (
        chapter10_dir
        / "98_gpt2_medium_sst2_batch8"
    )

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            "SST-2が見つかりません。\n"
            f"train: {train_path}\n"
            f"dev:   {dev_path}"
        )

    # --------------------------------------------------------
    # SST-2を読み込む
    # --------------------------------------------------------

    train_texts, train_labels = load_sst2(
        train_path
    )

    dev_texts, dev_labels = load_sst2(
        dev_path
    )

    if DEBUG:
        train_texts = train_texts[:512]
        train_labels = train_labels[:512]

        dev_texts = dev_texts[:100]
        dev_labels = dev_labels[:100]

    print(f"train size: {len(train_texts)}")
    print(f"dev size:   {len(dev_texts)}")

    # --------------------------------------------------------
    # CPU / GPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # GPU使用時だけmixed precisionを有効にする
    use_amp = device.type == "cuda"

    print(f"device: {device}")
    print(f"train batch size: {TRAIN_BATCH_SIZE}")
    print("gradient accumulation: disabled")
    print("gradient checkpointing: disabled")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # GPT-2 language model
    # --------------------------------------------------------

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # gradient checkpointingは使わない
    # 学習時は生成用キャッシュを使わない
    model.config.use_cache = False

    model.to(device)

    # --------------------------------------------------------
    # Dataset / DataLoader
    # --------------------------------------------------------

    train_dataset = SentimentFineTuningDataset(
        train_texts,
        train_labels,
        tokenizer,
        MAX_LENGTH,
    )

    dev_dataset = SentimentFineTuningDataset(
        dev_texts,
        dev_labels,
        tokenizer,
        MAX_LENGTH,
    )

    collator = CausalLMCollator(
        tokenizer.pad_token_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    # --------------------------------------------------------
    # Optimizer / mixed precision
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # mixed precision時の勾配を安全に扱う
    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    best_dev_loss = float("inf")

    # ========================================================
    # ファインチューニング
    # ========================================================

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
        )

        for batch in progress_bar:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # 毎バッチの最初に、前回の勾配を消す
            optimizer.zero_grad()

            # 順伝播
            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                outputs = model(**batch)
                loss = outputs.loss

            # 逆伝播
            # この処理はファインチューニングに必須
            scaler.scale(loss).backward()

            # 勾配クリッピング前に、勾配のスケールを元へ戻す
            scaler.unscale_(optimizer)

            # 勾配が極端に大きくなることを防ぐ
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # この1バッチ分の勾配で、GPT-2の重みを更新する
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_train_loss = (
            total_train_loss
            / len(train_loader)
        )

        dev_loss = calculate_loss(
            model,
            dev_loader,
            device,
            use_amp,
        )

        print(
            f"epoch {epoch:2d} | "
            f"train loss: {average_train_loss:.4f} | "
            f"dev loss: {dev_loss:.4f}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss

            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            print(
                f"best model saved: {save_dir}"
            )

    # --------------------------------------------------------
    # 学習中モデルを削除し、最良モデルを読み直す
    # --------------------------------------------------------

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_model = GPT2LMHeadModel.from_pretrained(
        save_dir
    )

    best_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # 生成時にはキャッシュを使う
    best_model.config.use_cache = True

    best_model.to(device)

    # --------------------------------------------------------
    # 実際にpositive / negativeを生成して正解率を測る
    # --------------------------------------------------------

    dev_accuracy = evaluate_accuracy(
        best_model,
        tokenizer,
        dev_texts,
        dev_labels,
        device,
    )

    print(f"best dev loss: {best_dev_loss:.4f}")
    print(f"dev accuracy:  {dev_accuracy:.4f}")


if __name__ == "__main__":
    main()

#出力
# train size: 67349
# dev size:   872
# device: cuda
# train batch size: 8
# gradient accumulation: disabled
# gradient checkpointing: disabled
# Epoch 1/1: 100%|██████████████████████████████████████████████| 8419/8419 [27:30<00:00,  5.10it/s, loss=0.1024]
# Validation loss: 100%|█████████████████████████████████████████████████████████| 55/55 [00:04<00:00, 11.60it/s]
# epoch  1 | train loss: 0.1258 | dev loss: 0.1495
# Generation accuracy: 100%|███████████████████████████████████████████████████| 872/872 [00:28<00:00, 30.29it/s]
# best dev loss: 0.1495
# dev accuracy:  0.9128