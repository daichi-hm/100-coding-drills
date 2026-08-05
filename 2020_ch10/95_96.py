"""95・96: サブワード翻訳と学習過程の可視化。

このプログラムでは、SentencePieceでKFTTをサブワード化し、
PyTorchのTransformerを3エポック学習する。
各エポック終了時にtrain/devのlossとBLEUをWandBへ記録し、
学習後に任意文翻訳、test BLEU、ビーム幅ごとのdev BLEUを求める。

91.pyのTranslationTransformerと、93.pyの英文整形関数を再利用する。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import MeCab
import sentencepiece as spm
import torch
import unidic_lite
import wandb
from sacrebleu.metrics import BLEU
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# ============================================================
# 1. ファイルの場所
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROBLEM91_PATH = BASE_DIR / "91.py"
PROBLEM93_PATH = BASE_DIR / "93.py"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
TRAIN_JA_PATH = PROCESSED_DIR / "train.ja"
TRAIN_EN_PATH = PROCESSED_DIR / "train.en"
DEV_JA_PATH = PROCESSED_DIR / "dev.ja"
DEV_EN_PATH = PROCESSED_DIR / "dev.en"
TEST_JA_PATH = PROCESSED_DIR / "test.ja"
TEST_EN_PATH = PROCESSED_DIR / "test.en"

# SentencePiece本体と、SentencePieceで分割済みのデータを保存する。
SENTENCEPIECE_DIR = BASE_DIR / "data" / "sentencepiece"
JA_SP_PREFIX = SENTENCEPIECE_DIR / "kftt_ja"
EN_SP_PREFIX = SENTENCEPIECE_DIR / "kftt_en"
JA_SP_MODEL_PATH = JA_SP_PREFIX.with_suffix(".model")
EN_SP_MODEL_PATH = EN_SP_PREFIX.with_suffix(".model")

SUBWORD_DIR = BASE_DIR / "data" / "subword"
SUBWORD_PATHS = {
    "train_ja": SUBWORD_DIR / "train.ja",
    "train_en": SUBWORD_DIR / "train.en",
    "dev_ja": SUBWORD_DIR / "dev.ja",
    "dev_en": SUBWORD_DIR / "dev.en",
    "test_ja": SUBWORD_DIR / "test.ja",
    "test_en": SUBWORD_DIR / "test.en",
}

MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "95_96_best_model.pt"
HISTORY_PATH = MODEL_DIR / "95_96_history.json"

RESULT_DIR = BASE_DIR / "results"
SAMPLE_TRANSLATION_PATH = RESULT_DIR / "95_sample_translation.txt"
TEST_PREDICTION_PATH = RESULT_DIR / "95_test_predictions.en"
TEST_REFERENCE_PATH = RESULT_DIR / "95_test_references.en"
TEST_BLEU_PATH = RESULT_DIR / "95_test_bleu.json"
BEAM_CSV_PATH = RESULT_DIR / "95_beam_bleu.csv"
BEAM_PLOT_PATH = RESULT_DIR / "95_beam_bleu.png"


# ============================================================
# 2. 実験設定
# ============================================================

# SentencePieceの語彙数。日本語と英語で別々のモデルを学習する。
JA_VOCAB_SIZE = 16_000
EN_VOCAB_SIZE = 16_000

# Trueにすると、既存のSentencePieceモデルとサブワード化済みデータを作り直す。
FORCE_REBUILD_SUBWORDS = False

BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_SENTENCE_LENGTH = 256
GRADIENT_CLIP_NORM = 1.0
NUM_WORKERS = 2
RANDOM_SEED = 42

# train BLEUは計算時間が長いため、
# 訓練データから固定した1,000文で計算する。
TRAIN_BLEU_SAMPLE_SIZE = 1_000

# 1文の生成が終わらない場合に備えた上限。
MAX_GENERATION_TOKENS = 256

# 94番と同様に、複数のビーム幅で開発BLEUを比較する。
BEAM_WIDTHS = [1, 2, 4, 8, 16, 32, 64, 100]

# Noneなら全件を使用する。動作確認時のみ小さい整数へ変更する。
MAX_BEAM_SENTENCES: int | None = None

SAMPLE_JAPANESE_TEXT = "京都は日本の古都です。"
RUN_BEAM_SEARCH_EVALUATION = True

# GPUサーバからWandBへ接続できない場合は "offline" に変更できる。
WANDB_MODE = "online"
WANDB_PROJECT = "nlp100-machine-translation"
WANDB_RUN_NAME = "95-96-subword-transformer"


# ============================================================
# 3. 共通処理
# ============================================================


def set_random_seed(seed: int) -> None:
    """PythonとPyTorchの乱数シードを固定する。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(name: str, path: Path) -> ModuleType:
    """数字から始まる91.py・93.pyをファイルパスから読み込む。"""
    if not path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"ファイルを読み込めません: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_input_files() -> None:
    """実行前に、必要な90・91・93番の出力とコードを確認する。"""
    required_paths = [
        PROBLEM91_PATH,
        PROBLEM93_PATH,
        TRAIN_JA_PATH,
        TRAIN_EN_PATH,
        DEV_JA_PATH,
        DEV_EN_PATH,
        TEST_JA_PATH,
        TEST_EN_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"必要なファイルがありません:\n{missing_text}")


def save_lines(path: Path, lines: list[str]) -> None:
    """文字列のリストを1文1行で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# 4. SentencePieceモデルとサブワードデータの作成
# ============================================================


def train_sentencepiece_model(
    input_path: Path,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float,
) -> bool:
    """訓練データからSentencePieceモデルを作り、再作成したかを返す。"""
    model_path = model_prefix.with_suffix(".model")

    if model_path.exists() and not FORCE_REBUILD_SUBWORDS:
        print(f"SentencePieceモデルを再利用: {model_path}")
        return False

    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 特殊トークンIDは91番までと同じ値に固定する。
    # これにより、Transformer側で特殊IDを同じ考え方で扱える。
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        hard_vocab_limit=False,
    )
    print(f"SentencePieceモデルを作成: {model_path}")
    return True


def load_sentencepiece_processor(model_path: Path) -> spm.SentencePieceProcessor:
    """保存済みSentencePieceモデルを読み込む。"""
    if not model_path.exists():
        raise FileNotFoundError(f"SentencePieceモデルがありません: {model_path}")

    processor = spm.SentencePieceProcessor()
    if not processor.load(str(model_path)):
        raise RuntimeError(f"SentencePieceモデルを読み込めません: {model_path}")
    return processor


def convert_file_to_subwords(
    input_path: Path,
    output_path: Path,
    processor: spm.SentencePieceProcessor,
    force_rebuild: bool,
) -> None:
    """1行ごとの文をSentencePieceで分割して保存する。"""
    if output_path.exists() and not (FORCE_REBUILD_SUBWORDS or force_rebuild):
        print(f"サブワードデータを再利用: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(
            tqdm(input_file, desc=f"subword: {input_path.name}"),
            start=1,
        ):
            sentence = line.rstrip("\n")
            if not sentence.strip():
                raise ValueError(
                    f"{input_path}の{line_number}行目が空です。"
                )

            pieces = processor.encode(sentence, out_type=str)
            output_file.write(" ".join(pieces) + "\n")

    print(f"サブワードデータを作成: {output_path}")


def prepare_subword_data(
) -> tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
    """SentencePieceモデルとtrain/dev/testのサブワードデータを準備する。"""
    japanese_model_rebuilt = train_sentencepiece_model(
        input_path=TRAIN_JA_PATH,
        model_prefix=JA_SP_PREFIX,
        vocab_size=JA_VOCAB_SIZE,
        character_coverage=0.9995,
    )
    english_model_rebuilt = train_sentencepiece_model(
        input_path=TRAIN_EN_PATH,
        model_prefix=EN_SP_PREFIX,
        vocab_size=EN_VOCAB_SIZE,
        character_coverage=1.0,
    )

    japanese_sp = load_sentencepiece_processor(JA_SP_MODEL_PATH)
    english_sp = load_sentencepiece_processor(EN_SP_MODEL_PATH)

    conversion_jobs = [
        (
            TRAIN_JA_PATH,
            SUBWORD_PATHS["train_ja"],
            japanese_sp,
            japanese_model_rebuilt,
        ),
        (
            TRAIN_EN_PATH,
            SUBWORD_PATHS["train_en"],
            english_sp,
            english_model_rebuilt,
        ),
        (
            DEV_JA_PATH,
            SUBWORD_PATHS["dev_ja"],
            japanese_sp,
            japanese_model_rebuilt,
        ),
        (
            DEV_EN_PATH,
            SUBWORD_PATHS["dev_en"],
            english_sp,
            english_model_rebuilt,
        ),
        (
            TEST_JA_PATH,
            SUBWORD_PATHS["test_ja"],
            japanese_sp,
            japanese_model_rebuilt,
        ),
        (
            TEST_EN_PATH,
            SUBWORD_PATHS["test_en"],
            english_sp,
            english_model_rebuilt,
        ),
    ]

    for input_path, output_path, processor, force_rebuild in conversion_jobs:
        convert_file_to_subwords(
            input_path,
            output_path,
            processor,
            force_rebuild,
        )

    return japanese_sp, english_sp


def create_piece_to_id(
    processor: spm.SentencePieceProcessor,
) -> dict[str, int]:
    """SentencePieceの全pieceについて、pieceからIDを引く辞書を作る。"""
    return {
        processor.id_to_piece(piece_id): piece_id
        for piece_id in range(processor.get_piece_size())
    }


# ============================================================
# 5. DatasetとDataLoader
# ============================================================


class SubwordTranslationDataset(Dataset):
    """サブワード分割済みの日英対訳をID列として返すDataset。"""

    def __init__(
        self,
        japanese_path: Path,
        english_path: Path,
        japanese_piece_to_id: dict[str, int],
        english_piece_to_id: dict[str, int],
        max_sentence_length: int | None,
    ) -> None:
        with japanese_path.open("r", encoding="utf-8") as file:
            japanese_lines = file.read().splitlines()
        with english_path.open("r", encoding="utf-8") as file:
            english_lines = file.read().splitlines()

        if len(japanese_lines) != len(english_lines):
            raise ValueError(
                f"日英の行数が一致しません: "
                f"{len(japanese_lines):,} / {len(english_lines):,}"
            )

        self.japanese_piece_to_id = japanese_piece_to_id
        self.english_piece_to_id = english_piece_to_id

        self.japanese_unk_id = japanese_piece_to_id["<unk>"]
        self.english_unk_id = english_piece_to_id["<unk>"]
        self.japanese_bos_id = japanese_piece_to_id["<bos>"]
        self.japanese_eos_id = japanese_piece_to_id["<eos>"]
        self.english_bos_id = english_piece_to_id["<bos>"]
        self.english_eos_id = english_piece_to_id["<eos>"]

        self.sentence_pairs: list[tuple[str, str]] = []
        skipped_long_pairs = 0

        for japanese_line, english_line in zip(japanese_lines, english_lines):
            if not japanese_line.strip() or not english_line.strip():
                raise ValueError("サブワードデータに空文があります。")

            # 後で<bos>と<eos>を加えるため、文長に2を加える。
            japanese_length = len(japanese_line.split()) + 2
            english_length = len(english_line.split()) + 2

            if max_sentence_length is not None and (
                japanese_length > max_sentence_length
                or english_length > max_sentence_length
            ):
                skipped_long_pairs += 1
                continue

            self.sentence_pairs.append((japanese_line, english_line))

        print(
            f"[{japanese_path.stem}] 使用する文対={len(self):,}, "
            f"長文による除外={skipped_long_pairs:,}"
        )

    def __len__(self) -> int:
        return len(self.sentence_pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        japanese_line, english_line = self.sentence_pairs[index]

        japanese_ids = [
            self.japanese_piece_to_id.get(piece, self.japanese_unk_id)
            for piece in japanese_line.split()
        ]
        english_ids = [
            self.english_piece_to_id.get(piece, self.english_unk_id)
            for piece in english_line.split()
        ]

        japanese_ids = [self.japanese_bos_id, *japanese_ids, self.japanese_eos_id]
        english_ids = [self.english_bos_id, *english_ids, self.english_eos_id]

        return (
            torch.tensor(japanese_ids, dtype=torch.long),
            torch.tensor(english_ids, dtype=torch.long),
        )


def collate_translation_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    japanese_pad_id: int,
    english_pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """バッチ内の文長を<pad>でそろえる。"""
    japanese_sequences = [pair[0] for pair in batch]
    english_sequences = [pair[1] for pair in batch]

    japanese_batch = pad_sequence(
        japanese_sequences,
        batch_first=True,
        padding_value=japanese_pad_id,
    )
    english_batch = pad_sequence(
        english_sequences,
        batch_first=True,
        padding_value=english_pad_id,
    )
    return japanese_batch, english_batch


def create_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    japanese_pad_id: int,
    english_pad_id: int,
) -> DataLoader:
    """学習または評価用DataLoaderを作る。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=partial(
            collate_translation_batch,
            japanese_pad_id=japanese_pad_id,
            english_pad_id=english_pad_id,
        ),
    )


# ============================================================
# 6. 学習とloss評価
# ============================================================


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    english_pad_id: int,
) -> float:
    """訓練データを1周し、1トークンあたりの平均lossを返す。"""
    model.train()
    total_loss = 0.0
    total_target_tokens = 0

    for japanese_ids, english_ids in tqdm(data_loader, desc="train", leave=False):
        japanese_ids = japanese_ids.to(device, non_blocking=True)
        english_ids = english_ids.to(device, non_blocking=True)

        # Teacher Forcingでは、正解英文を1トークンずらして使う。
        # 例: 入力=<bos> I like / 正解=I like <eos>
        decoder_input_ids = english_ids[:, :-1]
        expected_ids = english_ids[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            japanese_ids=japanese_ids,
            decoder_input_ids=decoder_input_ids,
        )

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            expected_ids.reshape(-1),
        )
        if not torch.isfinite(loss):
            raise RuntimeError("lossがNaNまたは無限大になりました。")

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

        target_tokens = expected_ids.ne(english_pad_id).sum().item()
        total_loss += loss.item() * target_tokens
        total_target_tokens += target_tokens

    return total_loss / total_target_tokens


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    english_pad_id: int,
    description: str,
) -> float:
    """パラメータを更新せず、データ全体の平均lossを計算する。"""
    model.eval()
    total_loss = 0.0
    total_target_tokens = 0

    for japanese_ids, english_ids in tqdm(
        data_loader,
        desc=description,
        leave=False,
    ):
        japanese_ids = japanese_ids.to(device, non_blocking=True)
        english_ids = english_ids.to(device, non_blocking=True)

        decoder_input_ids = english_ids[:, :-1]
        expected_ids = english_ids[:, 1:]
        logits = model(japanese_ids, decoder_input_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            expected_ids.reshape(-1),
        )
        target_tokens = expected_ids.ne(english_pad_id).sum().item()
        total_loss += loss.item() * target_tokens
        total_target_tokens += target_tokens

    return total_loss / total_target_tokens


# ============================================================
# 7. Greedy DecodingとBLEU
# ============================================================


@torch.no_grad()
def encode_source_batch(
    model: nn.Module,
    source_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """日本語バッチをEncoderへ通し、Encoder出力とpadding maskを返す。"""
    source_padding_mask = source_ids.eq(model.japanese_pad_id)
    source_embeddings = (
        model.japanese_embedding(source_ids) * math.sqrt(model.d_model)
    )
    source_embeddings = model.positional_encoding(source_embeddings)

    memory = model.transformer.encoder(
        source_embeddings,
        src_key_padding_mask=source_padding_mask,
    )
    return memory, source_padding_mask


@torch.no_grad()
def greedy_decode_batch(
    model: nn.Module,
    source_ids: torch.Tensor,
    english_pad_id: int,
    english_bos_id: int,
    english_eos_id: int,
) -> torch.Tensor:
    """日本語の1バッチを、各時点で最大確率のトークンを選んで翻訳する。"""
    model.eval()
    memory, source_padding_mask = encode_source_batch(model, source_ids)
    batch_size = source_ids.size(0)

    generated_ids = torch.full(
        (batch_size, 1),
        fill_value=english_bos_id,
        dtype=torch.long,
        device=source_ids.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=source_ids.device)

    max_position_length = (
        model.positional_encoding.positional_encoding.size(1)
    )
    generation_limit = min(MAX_GENERATION_TOKENS, max_position_length - 1)

    for _ in range(generation_limit):
        target_embeddings = (
            model.english_embedding(generated_ids) * math.sqrt(model.d_model)
        )
        target_embeddings = model.positional_encoding(target_embeddings)
        causal_mask = model.create_causal_mask(
            target_length=generated_ids.size(1),
            device=generated_ids.device,
        )

        decoder_output = model.transformer.decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        next_token_logits = model.output_layer(decoder_output[:, -1]).clone()

        # <pad>と<bos>は、翻訳文の途中では生成させない。
        next_token_logits[:, english_pad_id] = -float("inf")
        next_token_logits[:, english_bos_id] = -float("inf")
        next_token_ids = torch.argmax(next_token_logits, dim=-1)

        # すでに<eos>へ到達した文は、その後も<eos>を追加して形をそろえる。
        next_token_ids = torch.where(
            finished,
            torch.full_like(next_token_ids, english_eos_id),
            next_token_ids,
        )
        generated_ids = torch.cat(
            [generated_ids, next_token_ids.unsqueeze(1)],
            dim=1,
        )

        finished |= next_token_ids.eq(english_eos_id)
        if bool(finished.all()):
            break

    return generated_ids


def ids_to_english_text(
    token_ids: list[int],
    english_sp: spm.SentencePieceProcessor,
    detokenize_english,
) -> str:
    """特殊トークンを除き、SentencePiece ID列を通常の英文へ戻す。"""
    special_ids = {
        english_sp.pad_id(),
        english_sp.bos_id(),
        english_sp.eos_id(),
    }
    content_ids = [token_id for token_id in token_ids if token_id not in special_ids]
    decoded_text = english_sp.decode_ids(content_ids)
    return detokenize_english(decoded_text.split())


@torch.no_grad()
def calculate_bleu(
    model: nn.Module,
    data_loader: DataLoader,
    english_sp: spm.SentencePieceProcessor,
    detokenize_english,
    device: torch.device,
    description: str,
    prediction_path: Path | None = None,
    reference_path: Path | None = None,
) -> float:
    """データ全体をGreedy Decodingし、corpus BLEUを計算する。"""
    hypotheses: list[str] = []
    references: list[str] = []

    for japanese_ids, english_ids in tqdm(data_loader, desc=description, leave=False):
        japanese_ids = japanese_ids.to(device, non_blocking=True)
        generated_ids = greedy_decode_batch(
            model=model,
            source_ids=japanese_ids,
            english_pad_id=english_sp.pad_id(),
            english_bos_id=english_sp.bos_id(),
            english_eos_id=english_sp.eos_id(),
        ).cpu().tolist()

        # 正解側のIDは学習用DataLoaderからそのまま利用できる。
        reference_ids = english_ids.tolist()

        hypotheses.extend(
            ids_to_english_text(ids, english_sp, detokenize_english)
            for ids in generated_ids
        )
        references.extend(
            ids_to_english_text(ids, english_sp, detokenize_english)
            for ids in reference_ids
        )

    if prediction_path is not None:
        save_lines(prediction_path, hypotheses)
    if reference_path is not None:
        save_lines(reference_path, references)

    return BLEU(tokenize="13a").corpus_score(hypotheses, [references]).score


# ============================================================
# 8. モデルの保存と読み込み
# ============================================================


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    dev_loss: float,
    train_bleu: float,
    dev_bleu: float,
    model_config: dict[str, int | float],
    japanese_vocab_size: int,
    english_vocab_size: int,
) -> None:
    """最良モデルと、後からモデルを再構成するための設定を保存する。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "train_bleu": train_bleu,
        "dev_bleu": dev_bleu,
        "model_config": model_config,
        "japanese_vocab_size": japanese_vocab_size,
        "english_vocab_size": english_vocab_size,
        "special_token_ids": {
            "japanese_pad_id": 0,
            "english_pad_id": 0,
        },
    }
    torch.save(checkpoint, BEST_MODEL_PATH)


def load_best_model(
    problem91: ModuleType,
    device: torch.device,
) -> nn.Module:
    """保存したbest checkpointからTransformerを復元する。"""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"学習済みモデルがありません: {BEST_MODEL_PATH}")

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    config = checkpoint["model_config"]

    model = problem91.TranslationTransformer(
        japanese_vocab_size=checkpoint["japanese_vocab_size"],
        english_vocab_size=checkpoint["english_vocab_size"],
        japanese_pad_id=checkpoint["special_token_ids"]["japanese_pad_id"],
        english_pad_id=checkpoint["special_token_ids"]["english_pad_id"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_position_length=config["max_position_length"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(
        f"best model: epoch={checkpoint['epoch']}, "
        f"dev BLEU={checkpoint['dev_bleu']:.2f}"
    )
    return model


def save_history(history: list[dict[str, int | float]]) -> None:
    """エポックごとの4指標をJSONへ保存する。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# 9. 任意文の翻訳
# ============================================================


def tokenize_raw_japanese(text: str) -> str:
    """任意入力の日本語を、90番と同様にMeCabで形態素分割する。"""
    tagger = MeCab.Tagger(f"-Owakati -d {unidic_lite.DICDIR}")
    tagger.parse("")
    tokenized = tagger.parse(text)
    if tokenized is None or not tokenized.strip():
        raise ValueError("日本語文を形態素分割できませんでした。")
    return tokenized.strip()


@torch.no_grad()
def translate_raw_japanese(
    model: nn.Module,
    text: str,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    detokenize_english,
    device: torch.device,
) -> str:
    """任意の日本語文をMeCab・SentencePieceで処理し、英語へ翻訳する。"""
    tokenized_text = tokenize_raw_japanese(text)
    source_ids = japanese_sp.encode(
        tokenized_text,
        out_type=int,
        add_bos=True,
        add_eos=True,
    )

    if len(source_ids) > MAX_SENTENCE_LENGTH:
        raise ValueError(f"入力文が長すぎます: {len(source_ids)}トークン")

    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)
    generated_ids = greedy_decode_batch(
        model=model,
        source_ids=source_tensor,
        english_pad_id=english_sp.pad_id(),
        english_bos_id=english_sp.bos_id(),
        english_eos_id=english_sp.eos_id(),
    )[0].cpu().tolist()

    return ids_to_english_text(generated_ids, english_sp, detokenize_english)


# ============================================================
# 10. ビーム探索
# ============================================================


@dataclass
class Beam:
    """ビーム探索で保持する1つの生成途中候補。"""

    token_ids: list[int]
    sum_log_probability: float
    finished: bool

    @property
    def score(self) -> float:
        # <bos>はモデルが生成したトークンではないため、長さから除く。
        generated_length = max(1, len(self.token_ids) - 1)
        return self.sum_log_probability / generated_length


@torch.no_grad()
def predict_next_log_probabilities(
    model: nn.Module,
    beams: list[Beam],
    memory: torch.Tensor,
    source_padding_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """各ビーム候補について、次の全英語pieceの対数確率を返す。"""
    decoder_input_ids = torch.tensor(
        [beam.token_ids for beam in beams],
        dtype=torch.long,
        device=device,
    )
    target_embeddings = (
        model.english_embedding(decoder_input_ids) * math.sqrt(model.d_model)
    )
    target_embeddings = model.positional_encoding(target_embeddings)
    causal_mask = model.create_causal_mask(
        target_length=decoder_input_ids.size(1),
        device=device,
    )

    # 日本語文は1つだが英語候補は複数あるため、Encoder出力とmaskの
    # バッチ次元を候補数に合わせる。expandは実データを複製せずに参照を広げる。
    batch_size = len(beams)
    expanded_memory = memory.expand(batch_size, -1, -1)
    expanded_source_mask = source_padding_mask.expand(batch_size, -1)

    decoder_output = model.transformer.decoder(
        tgt=target_embeddings,
        memory=expanded_memory,
        tgt_mask=causal_mask,
        memory_key_padding_mask=expanded_source_mask,
    )
    logits = model.output_layer(decoder_output[:, -1])
    return torch.log_softmax(logits.float(), dim=-1)


@torch.no_grad()
def translate_with_beam_search(
    model: nn.Module,
    tokenized_japanese: str,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    detokenize_english,
    beam_width: int,
    device: torch.device,
) -> str:
    """形態素分割済み日本語1文をビーム探索で翻訳する。"""
    source_ids = japanese_sp.encode(
        tokenized_japanese,
        out_type=int,
        add_bos=True,
        add_eos=True,
    )
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)
    memory, source_padding_mask = encode_source_batch(model, source_tensor)

    pad_id = english_sp.pad_id()
    bos_id = english_sp.bos_id()
    eos_id = english_sp.eos_id()

    # 開始時点では、<bos>だけを持つ候補が1つある。
    beams = [Beam(token_ids=[bos_id], sum_log_probability=0.0, finished=False)]

    max_position_length = (
        model.positional_encoding.positional_encoding.size(1)
    )
    generation_limit = min(MAX_GENERATION_TOKENS, max_position_length - 1)

    for _ in range(generation_limit):
        active_beams = [beam for beam in beams if not beam.finished]
        if not active_beams:
            break

        log_probabilities = predict_next_log_probabilities(
            model=model,
            beams=active_beams,
            memory=memory,
            source_padding_mask=source_padding_mask,
            device=device,
        )
        log_probabilities[:, pad_id] = -float("inf")
        log_probabilities[:, bos_id] = -float("inf")

        # 各行には「その親候補の次に各pieceを生成する対数確率」が入る。
        # 親候補がそれまでに持つ合計対数確率を、その行の全pieceへ加える。
        previous_sums = torch.tensor(
            [beam.sum_log_probability for beam in active_beams],
            device=device,
        ).unsqueeze(1)
        candidate_sums = previous_sums + log_probabilities

        # 2次元の「親候補×次piece」を1次元に並べ、全組合せから上位を選ぶ。
        vocabulary_size = candidate_sums.size(1)
        top_values, top_indices = torch.topk(
            candidate_sums.flatten(),
            k=min(beam_width, candidate_sums.numel()),
        )

        # flatten後の番号を、親候補番号と追加piece IDへ戻す。
        parent_indices = top_indices // vocabulary_size
        next_token_ids = top_indices % vocabulary_size

        # 完成済み候補はそのまま候補集合に残す。
        candidates = [beam for beam in beams if beam.finished]
        for parent_index, token_id, sum_log_probability in zip(
            parent_indices.tolist(),
            next_token_ids.tolist(),
            top_values.tolist(),
        ):
            parent = active_beams[parent_index]
            candidates.append(
                Beam(
                    token_ids=parent.token_ids + [token_id],
                    sum_log_probability=sum_log_probability,
                    finished=(token_id == eos_id),
                )
            )

        # 文長による不利を抑えるため、平均対数確率で上位候補を残す。
        beams = sorted(
            candidates,
            key=lambda beam: beam.score,
            reverse=True,
        )[:beam_width]

        if all(beam.finished for beam in beams):
            break

    best_beam = max(beams, key=lambda beam: beam.score)
    return ids_to_english_text(best_beam.token_ids, english_sp, detokenize_english)


def load_parallel_lines(
    japanese_path: Path,
    english_path: Path,
    max_sentences: int | None,
) -> tuple[list[str], list[str]]:
    """日英データを読み込み、同じ行番号の対訳を返す。"""
    japanese_lines = japanese_path.read_text(encoding="utf-8").splitlines()
    english_lines = english_path.read_text(encoding="utf-8").splitlines()

    if len(japanese_lines) != len(english_lines):
        raise ValueError("日英データの行数が一致しません。")

    if max_sentences is not None:
        japanese_lines = japanese_lines[:max_sentences]
        english_lines = english_lines[:max_sentences]

    return japanese_lines, english_lines


def save_beam_results(results: list[tuple[int, float]]) -> None:
    """ビーム幅ごとのBLEUをCSVと折れ線グラフへ保存する。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with BEAM_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["beam_width", "bleu"])
        writer.writerows(results)

    widths = [width for width, _ in results]
    scores = [score for _, score in results]

    plt.figure(figsize=(8, 5))
    plt.plot(widths, scores, marker="o")
    plt.xlabel("Beam width")
    plt.ylabel("BLEU")
    plt.title("Beam width and BLEU on KFTT development set")
    plt.xticks(widths)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BEAM_PLOT_PATH, dpi=150)
    plt.close()


@torch.no_grad()
def evaluate_beam_widths(
    model: nn.Module,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    detokenize_english,
    device: torch.device,
) -> None:
    """ビーム幅を変え、開発データ上のBLEUを比較する。"""
    japanese_lines, tokenized_references = load_parallel_lines(
        DEV_JA_PATH,
        DEV_EN_PATH,
        MAX_BEAM_SENTENCES,
    )
    references = [
        detokenize_english(reference.split())
        for reference in tokenized_references
    ]

    bleu = BLEU(tokenize="13a")
    results: list[tuple[int, float]] = []

    for beam_width in BEAM_WIDTHS:
        hypotheses = [
            translate_with_beam_search(
                model=model,
                tokenized_japanese=japanese_line,
                japanese_sp=japanese_sp,
                english_sp=english_sp,
                detokenize_english=detokenize_english,
                beam_width=beam_width,
                device=device,
            )
            for japanese_line in tqdm(
                japanese_lines,
                desc=f"beam={beam_width}",
            )
        ]

        score = bleu.corpus_score(hypotheses, [references]).score
        results.append((beam_width, score))
        save_beam_results(results)
        print(f"beam width={beam_width:3d}, dev BLEU={score:.2f}")


# ============================================================
# 11. main
# ============================================================


def main() -> None:
    check_input_files()
    set_random_seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPU上で実行してください。")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    problem91 = load_module("problem91", PROBLEM91_PATH)
    problem93 = load_module("problem93", PROBLEM93_PATH)
    japanese_sp, english_sp = prepare_subword_data()

    japanese_piece_to_id = create_piece_to_id(japanese_sp)
    english_piece_to_id = create_piece_to_id(english_sp)

    train_dataset = SubwordTranslationDataset(
        SUBWORD_PATHS["train_ja"],
        SUBWORD_PATHS["train_en"],
        japanese_piece_to_id,
        english_piece_to_id,
        MAX_SENTENCE_LENGTH,
    )
    dev_dataset = SubwordTranslationDataset(
        SUBWORD_PATHS["dev_ja"],
        SUBWORD_PATHS["dev_en"],
        japanese_piece_to_id,
        english_piece_to_id,
        MAX_SENTENCE_LENGTH,
    )
    test_dataset = SubwordTranslationDataset(
        SUBWORD_PATHS["test_ja"],
        SUBWORD_PATHS["test_en"],
        japanese_piece_to_id,
        english_piece_to_id,
        MAX_SENTENCE_LENGTH,
    )

    japanese_pad_id = japanese_sp.pad_id()
    english_pad_id = english_sp.pad_id()

    # 学習用はshuffle=True、BLEU・loss評価用は順序を固定する。
    train_loader = create_data_loader(
        train_dataset,
        BATCH_SIZE,
        True,
        japanese_pad_id,
        english_pad_id,
    )
    # train BLEUの計算には、訓練データ全体ではなく
    # 固定した1,000文のサンプルを使う。
    train_bleu_sample_size = min(
        TRAIN_BLEU_SAMPLE_SIZE,
        len(train_dataset),
    )

    # 毎エポックで同じ文を評価できるように、
    # 専用の乱数生成器へ固定シードを設定する。
    train_bleu_generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_bleu_indices = torch.randperm(
        len(train_dataset),
        generator=train_bleu_generator,
    )[:train_bleu_sample_size].tolist()

    train_bleu_dataset = Subset(
        train_dataset,
        train_bleu_indices,
    )

    train_eval_loader = create_data_loader(
        train_bleu_dataset,
        BATCH_SIZE,
        False,
        japanese_pad_id,
        english_pad_id,
    )

    print(
        f"train BLEU評価用サンプル="
        f"{len(train_bleu_dataset):,}文"
    )
    dev_loader = create_data_loader(
        dev_dataset,
        BATCH_SIZE,
        False,
        japanese_pad_id,
        english_pad_id,
    )
    test_loader = create_data_loader(
        test_dataset,
        BATCH_SIZE,
        False,
        japanese_pad_id,
        english_pad_id,
    )

    model_config = {
        "d_model": problem91.D_MODEL,
        "nhead": problem91.NHEAD,
        "num_encoder_layers": problem91.NUM_ENCODER_LAYERS,
        "num_decoder_layers": problem91.NUM_DECODER_LAYERS,
        "dim_feedforward": problem91.DIM_FEEDFORWARD,
        "dropout": problem91.DROPOUT,
        "max_position_length": problem91.MAX_POSITION_LENGTH,
    }

    model = problem91.TranslationTransformer(
        japanese_vocab_size=japanese_sp.get_piece_size(),
        english_vocab_size=english_sp.get_piece_size(),
        japanese_pad_id=japanese_pad_id,
        english_pad_id=english_pad_id,
        **model_config,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=english_pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: list[dict[str, int | float]] = []
    best_dev_bleu = -float("inf")

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        mode=WANDB_MODE,
        config={
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "train_bleu_sample_size": len(train_bleu_dataset),
            "japanese_vocab_size": japanese_sp.get_piece_size(),
            "english_vocab_size": english_sp.get_piece_size(),
            **model_config,
        },
    )

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                english_pad_id,
            )
            dev_loss = evaluate_loss(
                model,
                dev_loader,
                criterion,
                device,
                english_pad_id,
                description="dev loss",
            )

            # 96番の指定に合わせ、train/devの両方でBLEUを計測する。
            # train BLEUもサンプリングせず、長文除外後の訓練データ全体を使う。
            train_bleu = calculate_bleu(
                model,
                train_eval_loader,
                english_sp,
                problem93.detokenize_english,
                device,
                description="train BLEU",
            )
            dev_bleu = calculate_bleu(
                model,
                dev_loader,
                english_sp,
                problem93.detokenize_english,
                device,
                description="dev BLEU",
            )

            epoch_result = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bleu": train_bleu,
                "dev_loss": dev_loss,
                "dev_bleu": dev_bleu,
            }
            history.append(epoch_result)
            save_history(history)
            wandb.log(epoch_result, step=epoch)

            print(
                f"epoch {epoch}/{NUM_EPOCHS} | "
                f"train loss={train_loss:.4f}, train BLEU={train_bleu:.2f} | "
                f"dev loss={dev_loss:.4f}, dev BLEU={dev_bleu:.2f}"
            )

            if dev_bleu > best_dev_bleu:
                best_dev_bleu = dev_bleu
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    dev_loss=dev_loss,
                    train_bleu=train_bleu,
                    dev_bleu=dev_bleu,
                    model_config=model_config,
                    japanese_vocab_size=japanese_sp.get_piece_size(),
                    english_vocab_size=english_sp.get_piece_size(),
                )
                print(f"best modelを保存: {BEST_MODEL_PATH}")
    finally:
        wandb.finish()

    best_model = load_best_model(problem91, device)

    sample_translation = translate_raw_japanese(
        model=best_model,
        text=SAMPLE_JAPANESE_TEXT,
        japanese_sp=japanese_sp,
        english_sp=english_sp,
        detokenize_english=problem93.detokenize_english,
        device=device,
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_TRANSLATION_PATH.write_text(
        f"Japanese: {SAMPLE_JAPANESE_TEXT}\nEnglish: {sample_translation}\n",
        encoding="utf-8",
    )
    print(f"sample: {SAMPLE_JAPANESE_TEXT} -> {sample_translation}")

    test_bleu = calculate_bleu(
        model=best_model,
        data_loader=test_loader,
        english_sp=english_sp,
        detokenize_english=problem93.detokenize_english,
        device=device,
        description="test BLEU",
        prediction_path=TEST_PREDICTION_PATH,
        reference_path=TEST_REFERENCE_PATH,
    )
    TEST_BLEU_PATH.write_text(
        json.dumps({"test_bleu": test_bleu}, indent=2),
        encoding="utf-8",
    )
    print(f"test BLEU={test_bleu:.2f}")

    if RUN_BEAM_SEARCH_EVALUATION:
        evaluate_beam_widths(
            model=best_model,
            japanese_sp=japanese_sp,
            english_sp=english_sp,
            detokenize_english=problem93.detokenize_english,
            device=device,
        )

    print("\n保存先")
    print("best model:", BEST_MODEL_PATH)
    print("history:", HISTORY_PATH)
    print("test BLEU:", TEST_BLEU_PATH)
    if RUN_BEAM_SEARCH_EVALUATION:
        print("beam CSV:", BEAM_CSV_PATH)
        print("beam plot:", BEAM_PLOT_PATH)


if __name__ == "__main__":
    main()
