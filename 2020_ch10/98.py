"""98: JParaCrawlを利用したKFTTへのドメイン適応。
    JESCやJParaCrawlなどの翻訳データを活用し、
    KFTTのテストデータの性能向上を試みる。

このコードでは、次の手順で実験する。

1. JParaCrawl v3.0から日英対訳の一部を抽出する。
2. 90番と同じ考え方で、
   日本語をMeCab、英語をMosesTokenizerで単語分割する。
3. 95・96番で作ったKFTTのSentencePieceモデルを使い、
   JParaCrawlも同じサブワード空間へ変換する。
4. 新しく初期化したTransformerをJParaCrawlで事前学習する。
5. 97番と同じハイパーパラメータ・最大エポック数で、
   KFTT trainを使ってファインチューニングする。
6. KFTT dev BLEUが最大のモデルを保存する。
7. KFTT test BLEUを計算し、97番の結果と比較する。

 JParaCrawl -> KFTT

    の順で学習する。これにより、外部対訳データによる事前学習と
    KFTTへのドメイン適応という流れを分かりやすくする。

JParaCrawlの準備:
    JParaCrawlの利用規約を公式ページで確認したうえで、
    v3.0の日英対訳アーカイブを次へ保存する。

        data/download/en-ja.tar.gz

    例:
        mkdir -p data/download
        wget -O data/download/en-ja.tar.gz \
          https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/en/3.0/bitext/en-ja.tar.gz

    アーカイブは非常に大きいため、GitHubへ追加しないこと。
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import io
import json
import random
import re
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import MeCab
import sentencepiece as spm
import torch
import unidic_lite
import wandb
from sacremoses import MosesTokenizer
from torch import nn
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


# ============================================================
# 1. ファイルの場所
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# これまでに作ったコードを再利用する。
PROBLEM91_PATH = BASE_DIR / "91.py"
PROBLEM93_PATH = BASE_DIR / "93.py"
PROBLEM95_96_PATH = BASE_DIR / "95_96.py"

# 97番の最良モデルと評価結果。
# モデル構造・最良ハイパーパラメータ・基準BLEUをここから取得する。
PROBLEM97_MODEL_PATH = BASE_DIR / "models" / "97_best_model.pt"
PROBLEM97_TEST_RESULT_PATH = BASE_DIR / "results" / "97_test_bleu.json"
PROBLEM97_HISTORY_PATH = BASE_DIR / "results" / "97_final_history.json"

# JParaCrawl公式アーカイブ。
JPARACRAW_URL = (
    "https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/"
    "release/en/3.0/bitext/en-ja.tar.gz"
)
JPARACRAW_ARCHIVE_PATH = BASE_DIR / "data" / "download" / "en-ja.tar.gz"

# JParaCrawlを抽出・トークン化・サブワード化した結果。
EXTERNAL_DIR = BASE_DIR / "data" / "external" / "jparacrawl"
EXTERNAL_JA_PATH = EXTERNAL_DIR / "train.subword.ja"
EXTERNAL_EN_PATH = EXTERNAL_DIR / "train.subword.en"
EXTERNAL_METADATA_PATH = EXTERNAL_DIR / "metadata.json"
EXTERNAL_PREVIEW_PATH = EXTERNAL_DIR / "preview.json"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

# JParaCrawlで事前学習した直後のモデル。
# ここを保存しておくと、KFTTファインチューニングだけをやり直せる。
PRETRAINED_MODEL_PATH = MODEL_DIR / "98_jparacrawl_pretrained.pt"

# KFTT dev BLEUが最大だった98番の最終モデル。
BEST_MODEL_PATH = MODEL_DIR / "98_best_model.pt"

# KFTTファインチューニング中の履歴。
HISTORY_PATH = RESULT_DIR / "98_history.json"

# KFTT testの最終結果。
TEST_RESULT_PATH = RESULT_DIR / "98_test_bleu.json"
TEST_PREDICTION_PATH = RESULT_DIR / "98_test_predictions.en"
TEST_REFERENCE_PATH = RESULT_DIR / "98_test_references.en"

# 勉強会で見せやすい比較図。
TEST_COMPARISON_PLOT_PATH = RESULT_DIR / "98_test_bleu_comparison.png"
DEV_CURVE_PLOT_PATH = RESULT_DIR / "98_dev_bleu_comparison.png"


# ============================================================
# 2. 実験設定
# ============================================================

RANDOM_SEED = 42
MAX_SENTENCE_LENGTH = 256

# -------------------- JParaCrawlの前処理 --------------------

# JParaCrawl全件を学習に使うと非常に重いため、
# 品質条件を満たした文対から固定した最大100万文を使う。
TARGET_EXTERNAL_TRAIN_SENTENCES = 1_000_000

# JParaCrawlは巨大なので、全行のうちハッシュ値で選ばれた一部だけを
# MeCab・MosesTokenizer・SentencePieceへ通す。
# 同じ文は常に同じ判定になるため、再現性がある。
# 0.10は約10%を前処理候補にするという意味であり、
# 品質・文長フィルタ後の件数はこれより少なくなる。
JPARACRAW_HASH_SAMPLE_RATE = 0.10

# JParaCrawlのアラインメント品質スコアがこの値未満の文対を除外する。
# 公式ファイルの通常形式では、第3列にスコアが入っている。
MIN_ALIGNMENT_SCORE = 0.70

# 極端に長い生文を、重い形態素解析へ入れる前に除外する。
MAX_RAW_CHARACTERS = 1_000

# 日英の文字数比が極端な文対は、対訳でない可能性があるため除外する。
MAX_CHARACTER_LENGTH_RATIO = 4.0

# 前処理済みファイルが存在するときは再利用する。
# 条件を変えて作り直す場合だけTrueへ変更する。
FORCE_REBUILD_EXTERNAL_DATA = False

# -------------------- 学習設定 --------------------

# JParaCrawlによる事前学習は1エポックとする。
EXTERNAL_PRETRAIN_EPOCHS = 1

# 97番の最終学習と公平に比較するため、
# KFTTファインチューニングも最大5エポックにする。
KFTT_FINETUNE_EPOCHS = 5

# 97番と同じ考え方で、dev BLEUが改善しない状態が続けば終了する。
EARLY_STOPPING_PATIENCE = 2

# dev/test評価では勾配を保存しないため、学習より大きい値を使える。
# GPUメモリ不足になる場合は16へ下げる。
EVAL_BATCH_SIZE = 32

# 事前学習checkpointがあれば再利用する。
# JParaCrawl事前学習からやり直したい場合はFalseへ変更する。
REUSE_PRETRAINED_CHECKPOINT = True

# -------------------- 動作確認 --------------------

# Trueではデータ件数・エポック数を小さくして、最後まで動くかだけ確認する。
# 正式な98番の結果を出すときは必ずFalseにする。
DEBUG = False
DEBUG_EXTERNAL_SENTENCES = 10_000
DEBUG_KFTT_SENTENCES = 10_000
DEBUG_FINETUNE_EPOCHS = 1

# -------------------- W&B --------------------

WANDB_MODE = "online"
WANDB_PROJECT = "nlp100-machine-translation"
WANDB_RUN_NAME = "98-jparacrawl-domain-adaptation"


# ============================================================
# 3. 共通の小さな関数
# ============================================================


def set_random_seed(seed: int) -> None:
    """PythonとPyTorchの乱数シードを固定する。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_module(name: str, path: Path) -> ModuleType:
    """数字から始まるPythonファイルを、ファイルパスから読み込む。"""
    if not path.exists():
        raise FileNotFoundError(f"必要なファイルがありません: {path}")

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Pythonファイルを読み込めません: {path}")

    module = importlib.util.module_from_spec(spec)

    # @dataclassなどは、定義元モジュールをsys.modulesから参照する。
    # そのため、ファイルを実行する前に登録しておく。
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # 読み込みに失敗した不完全なモジュールを残さない。
        sys.modules.pop(name, None)
        raise

    return module


def cleanup_gpu() -> None:
    """不要なオブジェクトとGPUキャッシュを解放する。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(path: Path, data: object) -> None:
    """辞書やリストを読みやすいJSON形式で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path) -> object:
    """JSONファイルを読み込む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def external_metadata_is_compatible(metadata: object) -> bool:
    """前処理済みJParaCrawlが現在の設定で作られたか確認する。"""
    if not isinstance(metadata, dict):
        return False

    return (
        bool(metadata.get("debug", False)) == DEBUG
        and float(metadata.get("minimum_alignment_score", -1.0))
        == MIN_ALIGNMENT_SCORE
        and float(metadata.get("hash_sample_rate", -1.0))
        == JPARACRAW_HASH_SAMPLE_RATE
        and int(metadata.get("max_sentence_length", -1))
        == MAX_SENTENCE_LENGTH
    )


def pretrained_checkpoint_is_compatible() -> bool:
    """保存済み事前学習モデルを現在のDEBUG設定で再利用できるか確認する。"""
    if not REUSE_PRETRAINED_CHECKPOINT or not PRETRAINED_MODEL_PATH.exists():
        return False

    checkpoint = torch.load(
        PRETRAINED_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    return bool(checkpoint.get("debug", False)) == DEBUG


def prepared_external_data_is_compatible() -> bool:
    """前処理済み外部データ3ファイルが揃い、設定も一致するか確認する。"""
    if not (
        EXTERNAL_JA_PATH.exists()
        and EXTERNAL_EN_PATH.exists()
        and EXTERNAL_METADATA_PATH.exists()
    ):
        return False

    try:
        metadata = load_json(EXTERNAL_METADATA_PATH)
    except (OSError, json.JSONDecodeError):
        return False

    return external_metadata_is_compatible(metadata)


def check_required_files() -> None:
    """コード・97番の結果・JParaCrawlの準備状況を確認する。"""
    required_paths = [
        PROBLEM91_PATH,
        PROBLEM93_PATH,
        PROBLEM95_96_PATH,
        PROBLEM97_MODEL_PATH,
        PROBLEM97_TEST_RESULT_PATH,
        PROBLEM97_HISTORY_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "98番に必要なファイルがありません。\n"
            "先に97番を完了し、91.py・93.py・95_96.pyも"
            f"同じフォルダに置いてください。\n\n{missing_text}"
        )

    # 現在の設定と互換性がある前処理済みデータまたは
    # 事前学習checkpointがなければ、元アーカイブが必要になる。
    if (
        not prepared_external_data_is_compatible()
        and not pretrained_checkpoint_is_compatible()
    ):
        if not JPARACRAW_ARCHIVE_PATH.exists():
            raise FileNotFoundError(
                "JParaCrawlのアーカイブがありません。\n"
                f"保存先: {JPARACRAW_ARCHIVE_PATH}\n\n"
                "公式利用規約を確認してから、次のURLのファイルを"
                f"上記の場所へ保存してください。\n{JPARACRAW_URL}"
            )


def create_fixed_subset(
    dataset: Dataset,
    sample_size: int | None,
    seed: int,
) -> Dataset:
    """同じ乱数シードで固定された部分データを作る。"""
    if sample_size is None or sample_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:sample_size].tolist()

    return Subset(dataset, indices)


# ============================================================
# 4. 97番の設定と基準値を読み込む
# ============================================================


def load_problem97_information() -> tuple[dict[str, object], float]:
    """97番のcheckpointとtest BLEUを読み込む。"""
    checkpoint = torch.load(
        PROBLEM97_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    test_result = load_json(PROBLEM97_TEST_RESULT_PATH)
    if not isinstance(test_result, dict) or "test_bleu" not in test_result:
        raise ValueError(
            f"{PROBLEM97_TEST_RESULT_PATH}にtest_bleuがありません。"
        )

    baseline_test_bleu = float(test_result["test_bleu"])

    if "model_config" not in checkpoint:
        raise ValueError("97番checkpointにmodel_configがありません。")
    if "hyperparameters" not in checkpoint:
        raise ValueError("97番checkpointにhyperparametersがありません。")

    return checkpoint, baseline_test_bleu


def get_best_hyperparameters(
    problem97_checkpoint: dict[str, object],
) -> tuple[int, float, str]:
    """97番の最良バッチサイズ・学習率・Optimizerを取り出す。"""
    hyperparameters = problem97_checkpoint["hyperparameters"]
    if not isinstance(hyperparameters, dict):
        raise TypeError("97番のhyperparametersが辞書ではありません。")

    batch_size = int(hyperparameters["batch_size"])
    learning_rate = float(hyperparameters["learning_rate"])

    # 97.pyではoptimizer_nameというキーで保存している。
    # 過去版との互換性のためoptimizerも確認する。
    optimizer_name_value = hyperparameters.get(
        "optimizer_name",
        hyperparameters.get("optimizer"),
    )
    if optimizer_name_value is None:
        raise KeyError("97番のOptimizer名がcheckpointにありません。")

    return batch_size, learning_rate, str(optimizer_name_value)


def create_model_from_problem97_config(
    problem91: ModuleType,
    problem97_checkpoint: dict[str, object],
    device: torch.device,
) -> nn.Module:
    """97番と同じ構造だが、重みは新しいTransformerを作る。"""
    config = problem97_checkpoint["model_config"]
    if not isinstance(config, dict):
        raise TypeError("97番のmodel_configが辞書ではありません。")

    special_token_ids = problem97_checkpoint["special_token_ids"]
    if not isinstance(special_token_ids, dict):
        raise TypeError("special_token_idsが辞書ではありません。")

    model = problem91.TranslationTransformer(
        japanese_vocab_size=int(
            problem97_checkpoint["japanese_vocab_size"]
        ),
        english_vocab_size=int(
            problem97_checkpoint["english_vocab_size"]
        ),
        japanese_pad_id=int(special_token_ids["japanese_pad_id"]),
        english_pad_id=int(special_token_ids["english_pad_id"]),
        d_model=int(config["d_model"]),
        nhead=int(config["nhead"]),
        num_encoder_layers=int(config["num_encoder_layers"]),
        num_decoder_layers=int(config["num_decoder_layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_position_length=int(config["max_position_length"]),
    )
    return model.to(device)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """97番で選ばれたAdam系Optimizerを作る。"""
    if optimizer_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

    if optimizer_name == "RAdam":
        return torch.optim.RAdam(model.parameters(), lr=learning_rate)

    raise ValueError(f"未対応のOptimizerです: {optimizer_name}")


# ============================================================
# 5. JParaCrawlの抽出・品質フィルタ・サブワード化
# ============================================================


JAPANESE_CHARACTER_PATTERN = re.compile(
    r"[ぁ-んァ-ヶ一-龠々〆ヵヶ]"
)
ENGLISH_LETTER_PATTERN = re.compile(r"[A-Za-z]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """改行・タブ・連続空白を1個の空白へまとめる。"""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def parse_jparacrawl_line(
    line: str,
) -> tuple[float, str, str] | None:
    """JParaCrawlの1行から、スコア・英語・日本語を取り出す。

    v3.0の代表的な形式:
        英語側ドメイン\t日本語側ドメイン\tスコア\t英語文\t日本語文

    旧版など列数が多少異なる場合にも対応するため、
    最後の2列を英語文・日本語文として扱う。
    """
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 4:
        return None

    english_text = normalize_text(fields[-2])
    japanese_text = normalize_text(fields[-1])

    try:
        alignment_score = float(fields[-3])
    except ValueError:
        # スコア列を持たない形式なら、品質スコアによる除外をしない。
        alignment_score = 1.0

    return alignment_score, english_text, japanese_text


def raw_pair_is_valid(english_text: str, japanese_text: str) -> bool:
    """形態素解析前に、明らかに不適切な文対を除外する。"""
    if not english_text or not japanese_text:
        return False

    if (
        len(english_text) > MAX_RAW_CHARACTERS
        or len(japanese_text) > MAX_RAW_CHARACTERS
    ):
        return False

    # 英語側に英字、日本語側に日本語文字が少なくとも1字必要。
    if ENGLISH_LETTER_PATTERN.search(english_text) is None:
        return False
    if JAPANESE_CHARACTER_PATTERN.search(japanese_text) is None:
        return False

    english_length = max(1, len(english_text.replace(" ", "")))
    japanese_length = max(1, len(japanese_text.replace(" ", "")))
    length_ratio = max(
        english_length / japanese_length,
        japanese_length / english_length,
    )

    return length_ratio <= MAX_CHARACTER_LENGTH_RATIO


def stable_hash_value(text: str) -> int:
    """Python実行ごとに変化しない64ビットのハッシュ値を返す。"""
    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def selected_by_hash(english_text: str, japanese_text: str) -> bool:
    """文対の約10%を、再現可能な方法で選ぶ。"""
    pair_text = f"{english_text}\t{japanese_text}"
    value = stable_hash_value(pair_text)

    # 64ビット整数が取り得る範囲のうち、
    # SAMPLE_RATEに対応する下側の範囲なら採用する。
    threshold = int(JPARACRAW_HASH_SAMPLE_RATE * (2**64))
    return value < threshold


def create_tokenizers() -> tuple[MeCab.Tagger, MosesTokenizer]:
    """90番と同じ種類の日本語・英語Tokenizerを準備する。"""
    japanese_tokenizer = MeCab.Tagger(
        f"-Owakati -d {unidic_lite.DICDIR}"
    )

    # MeCabの辞書読み込みエラーを早い段階で検出する。
    japanese_tokenizer.parse("")

    english_tokenizer = MosesTokenizer(lang="en")
    return japanese_tokenizer, english_tokenizer


def tokenize_and_encode_pair(
    japanese_text: str,
    english_text: str,
    japanese_tokenizer: MeCab.Tagger,
    english_tokenizer: MosesTokenizer,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
) -> tuple[list[str], list[str]] | None:
    """生の日英対訳を、KFTTと同じ前処理でサブワード化する。"""
    parsed_japanese = japanese_tokenizer.parse(japanese_text)
    if parsed_japanese is None:
        return None

    japanese_tokens = parsed_japanese.strip().split()
    english_tokens = english_tokenizer.tokenize(
        english_text,
        return_str=False,
        escape=False,
    )

    if not japanese_tokens or not english_tokens:
        return None

    # SentencePieceは95・96番でKFTTの分かち書き済み文から学習したため、
    # ここでも単語間を空白でつないだ文字列を入力する。
    japanese_pieces = japanese_sp.encode(
        " ".join(japanese_tokens),
        out_type=str,
    )
    english_pieces = english_sp.encode(
        " ".join(english_tokens),
        out_type=str,
    )

    # Datasetで<bos>と<eos>を加えるため、ここでは2を足して判定する。
    if (
        len(japanese_pieces) + 2 > MAX_SENTENCE_LENGTH
        or len(english_pieces) + 2 > MAX_SENTENCE_LENGTH
    ):
        return None

    return japanese_pieces, english_pieces


def find_parallel_text_member(
    archive: tarfile.TarFile,
) -> tarfile.TarInfo:
    """JParaCrawlアーカイブ内の対訳テキストを探す。"""
    text_members = [
        member
        for member in archive.getmembers()
        if member.isfile() and member.name.endswith(".txt")
    ]

    if not text_members:
        raise FileNotFoundError(
            "JParaCrawlアーカイブ内に.txtファイルがありません。"
        )

    # en-ja.bicleaner05.txtを優先する。
    for member in text_members:
        if member.name.endswith("en-ja.bicleaner05.txt"):
            return member

    # ファイル名が異なる版では、最も大きいテキストを対訳本体とみなす。
    return max(text_members, key=lambda member: member.size)


def prepare_external_subword_data(
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
) -> dict[str, int | float | str]:
    """JParaCrawlから固定サンプルを作り、サブワードファイルへ保存する。"""
    prepared_files_exist = (
        EXTERNAL_JA_PATH.exists()
        and EXTERNAL_EN_PATH.exists()
        and EXTERNAL_METADATA_PATH.exists()
    )

    if prepared_files_exist and not FORCE_REBUILD_EXTERNAL_DATA:
        metadata = load_json(EXTERNAL_METADATA_PATH)
        if external_metadata_is_compatible(metadata):
            print("JParaCrawl前処理済みデータを再利用します。")
            print("日本語:", EXTERNAL_JA_PATH)
            print("英語  :", EXTERNAL_EN_PATH)
            return metadata

        print(
            "既存のJParaCrawl前処理データは、現在のDEBUG・品質設定と"
            "一致しないため作り直します。"
        )

    if not JPARACRAW_ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"JParaCrawlアーカイブがありません: {JPARACRAW_ARCHIVE_PATH}"
        )

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    japanese_tokenizer, english_tokenizer = create_tokenizers()

    counters = {
        "scanned_lines": 0,
        "malformed_lines": 0,
        "low_score_pairs": 0,
        "invalid_raw_pairs": 0,
        "hash_not_selected_pairs": 0,
        "duplicate_pairs": 0,
        "tokenization_or_long_pairs": 0,
        "written_pairs": 0,
    }

    # 選ばれた文対だけのハッシュを保持し、重複を除く。
    seen_pair_hashes: set[int] = set()
    previews: list[dict[str, str | float]] = []

    # 書き込み途中で停止した場合に完成ファイルと区別できるよう、
    # 一時ファイルへ出力して最後にrenameする。
    temporary_ja_path = EXTERNAL_JA_PATH.with_suffix(".ja.part")
    temporary_en_path = EXTERNAL_EN_PATH.with_suffix(".en.part")

    print("JParaCrawlを読み込み、外部学習データを作成します。")
    print("archive:", JPARACRAW_ARCHIVE_PATH)

    try:
        with tarfile.open(JPARACRAW_ARCHIVE_PATH, mode="r:gz") as archive:
            member = find_parallel_text_member(archive)
            print("archive内の対訳ファイル:", member.name)

            binary_file = archive.extractfile(member)
            if binary_file is None:
                raise RuntimeError(
                    f"アーカイブ内のファイルを開けません: {member.name}"
                )

            with (
                io.TextIOWrapper(
                    binary_file,
                    encoding="utf-8",
                    errors="replace",
                ) as input_file,
                temporary_ja_path.open("w", encoding="utf-8") as ja_file,
                temporary_en_path.open("w", encoding="utf-8") as en_file,
            ):
                progress = tqdm(input_file, desc="prepare JParaCrawl")

                for line in progress:
                    counters["scanned_lines"] += 1

                    parsed = parse_jparacrawl_line(line)
                    if parsed is None:
                        counters["malformed_lines"] += 1
                        continue

                    alignment_score, english_text, japanese_text = parsed

                    if alignment_score < MIN_ALIGNMENT_SCORE:
                        counters["low_score_pairs"] += 1
                        continue

                    if not raw_pair_is_valid(english_text, japanese_text):
                        counters["invalid_raw_pairs"] += 1
                        continue

                    if not selected_by_hash(english_text, japanese_text):
                        counters["hash_not_selected_pairs"] += 1
                        continue

                    pair_hash = stable_hash_value(
                        f"{english_text}\t{japanese_text}"
                    )
                    if pair_hash in seen_pair_hashes:
                        counters["duplicate_pairs"] += 1
                        continue
                    seen_pair_hashes.add(pair_hash)

                    encoded = tokenize_and_encode_pair(
                        japanese_text=japanese_text,
                        english_text=english_text,
                        japanese_tokenizer=japanese_tokenizer,
                        english_tokenizer=english_tokenizer,
                        japanese_sp=japanese_sp,
                        english_sp=english_sp,
                    )
                    if encoded is None:
                        counters["tokenization_or_long_pairs"] += 1
                        continue

                    japanese_pieces, english_pieces = encoded
                    ja_file.write(" ".join(japanese_pieces) + "\n")
                    en_file.write(" ".join(english_pieces) + "\n")
                    counters["written_pairs"] += 1

                    if len(previews) < 5:
                        previews.append(
                            {
                                "alignment_score": alignment_score,
                                "raw_japanese": japanese_text,
                                "raw_english": english_text,
                                "subword_japanese": " ".join(japanese_pieces),
                                "subword_english": " ".join(english_pieces),
                            }
                        )

                    if counters["written_pairs"] % 1_000 == 0:
                        progress.set_postfix(
                            written=f"{counters['written_pairs']:,}"
                        )

                    # DEBUGでは全アーカイブを走査せず、必要件数で止める。
                    if (
                        DEBUG
                        and counters["written_pairs"]
                        >= DEBUG_EXTERNAL_SENTENCES
                    ):
                        break

        if counters["written_pairs"] == 0:
            raise RuntimeError(
                "品質条件を満たすJParaCrawl文対を作成できませんでした。"
            )

        temporary_ja_path.replace(EXTERNAL_JA_PATH)
        temporary_en_path.replace(EXTERNAL_EN_PATH)

    except Exception:
        temporary_ja_path.unlink(missing_ok=True)
        temporary_en_path.unlink(missing_ok=True)
        raise

    metadata: dict[str, int | float | str] = {
        "corpus": "JParaCrawl v3.0",
        "archive": str(JPARACRAW_ARCHIVE_PATH),
        "minimum_alignment_score": MIN_ALIGNMENT_SCORE,
        "hash_sample_rate": JPARACRAW_HASH_SAMPLE_RATE,
        "max_sentence_length": MAX_SENTENCE_LENGTH,
        "debug": DEBUG,
        **counters,
    }
    save_json(EXTERNAL_METADATA_PATH, metadata)
    save_json(EXTERNAL_PREVIEW_PATH, previews)

    print("\nJParaCrawl前処理結果")
    for key, value in counters.items():
        print(f"{key}: {value:,}")

    return metadata


# ============================================================
# 6. Datasetの準備
# ============================================================


def create_kftt_datasets(
    common: ModuleType,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
) -> tuple[Dataset, Dataset, Dataset]:
    """95・96番のサブワード化済みKFTTをDatasetへ変換する。"""
    japanese_piece_to_id = common.create_piece_to_id(japanese_sp)
    english_piece_to_id = common.create_piece_to_id(english_sp)

    train_dataset = common.SubwordTranslationDataset(
        common.SUBWORD_PATHS["train_ja"],
        common.SUBWORD_PATHS["train_en"],
        japanese_piece_to_id,
        english_piece_to_id,
        MAX_SENTENCE_LENGTH,
    )
    dev_dataset = common.SubwordTranslationDataset(
        common.SUBWORD_PATHS["dev_ja"],
        common.SUBWORD_PATHS["dev_en"],
        japanese_piece_to_id,
        english_piece_to_id,
        MAX_SENTENCE_LENGTH,
    )
    test_dataset = common.SubwordTranslationDataset(
        common.SUBWORD_PATHS["test_ja"],
        common.SUBWORD_PATHS["test_en"],
        japanese_piece_to_id,
        english_piece_to_id,
        MAX_SENTENCE_LENGTH,
    )

    if DEBUG:
        train_dataset = create_fixed_subset(
            train_dataset,
            DEBUG_KFTT_SENTENCES,
            RANDOM_SEED,
        )

    return train_dataset, dev_dataset, test_dataset


def create_external_dataset(
    common: ModuleType,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
) -> Dataset:
    """前処理済みJParaCrawlをDatasetへ変換し、最大件数を固定する。"""
    japanese_piece_to_id = common.create_piece_to_id(japanese_sp)
    english_piece_to_id = common.create_piece_to_id(english_sp)

    dataset = common.SubwordTranslationDataset(
        EXTERNAL_JA_PATH,
        EXTERNAL_EN_PATH,
        japanese_piece_to_id,
        english_piece_to_id,
        MAX_SENTENCE_LENGTH,
    )

    target_size = (
        DEBUG_EXTERNAL_SENTENCES
        if DEBUG
        else TARGET_EXTERNAL_TRAIN_SENTENCES
    )
    return create_fixed_subset(dataset, target_size, RANDOM_SEED)


# ============================================================
# 7. checkpointの保存・読み込み
# ============================================================


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """モデル重みをCPUへ移し、GPUに依存しない形で保存する。"""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def save_pretrained_checkpoint(
    model: nn.Module,
    epoch: int,
    train_loss: float,
    problem97_checkpoint: dict[str, object],
    external_train_sentences: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
) -> None:
    """JParaCrawl事前学習直後のモデルを保存する。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "stage": "jparacrawl_pretraining",
        "epoch": epoch,
        "train_loss": train_loss,
        "model_state_dict": cpu_state_dict(model),
        "model_config": problem97_checkpoint["model_config"],
        "japanese_vocab_size": problem97_checkpoint["japanese_vocab_size"],
        "english_vocab_size": problem97_checkpoint["english_vocab_size"],
        "special_token_ids": problem97_checkpoint["special_token_ids"],
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer_name": optimizer_name,
        },
        "external_corpus": "JParaCrawl v3.0",
        "external_train_sentences": external_train_sentences,
        "debug": DEBUG,
    }
    torch.save(checkpoint, PRETRAINED_MODEL_PATH)


def save_best_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    dev_loss: float,
    dev_bleu: float,
    problem97_checkpoint: dict[str, object],
    external_train_sentences: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
) -> None:
    """KFTT dev BLEUが最大の98番モデルを保存する。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "stage": "kftt_finetuning",
        "epoch": epoch,
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "dev_bleu": dev_bleu,
        "model_config": problem97_checkpoint["model_config"],
        "japanese_vocab_size": problem97_checkpoint["japanese_vocab_size"],
        "english_vocab_size": problem97_checkpoint["english_vocab_size"],
        "special_token_ids": problem97_checkpoint["special_token_ids"],
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer_name": optimizer_name,
        },
        "domain_adaptation": {
            "external_corpus": "JParaCrawl v3.0",
            "external_train_sentences": external_train_sentences,
            "external_pretrain_epochs": EXTERNAL_PRETRAIN_EPOCHS,
            "kftt_max_finetune_epochs": KFTT_FINETUNE_EPOCHS,
        },
    }
    torch.save(checkpoint, BEST_MODEL_PATH)


def load_model_state(
    model: nn.Module,
    checkpoint_path: Path,
) -> dict[str, object]:
    """保存済みcheckpointの重みをモデルへ読み込む。"""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


# ============================================================
# 8. JParaCrawlによる事前学習
# ============================================================


def pretrain_with_jparacrawl(
    model: nn.Module,
    common: ModuleType,
    external_dataset: Dataset,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    problem97_checkpoint: dict[str, object],
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    device: torch.device,
    run: wandb.sdk.wandb_run.Run,
) -> None:
    """新しいTransformerをJParaCrawlで事前学習する。"""
    if pretrained_checkpoint_is_compatible():
        load_model_state(model, PRETRAINED_MODEL_PATH)
        print("JParaCrawl事前学習済みモデルを再利用します:")
        print(PRETRAINED_MODEL_PATH)
        run.summary["pretraining_reused"] = True
        return

    print("\n" + "=" * 70)
    print("JParaCrawlによる事前学習")
    print(f"文対数: {len(external_dataset):,}")
    print(f"epoch: {EXTERNAL_PRETRAIN_EPOCHS}")
    print("=" * 70)

    train_loader = common.create_data_loader(
        external_dataset,
        batch_size,
        True,
        japanese_sp.pad_id(),
        english_sp.pad_id(),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=english_sp.pad_id())
    optimizer = create_optimizer(model, optimizer_name, learning_rate)

    try:
        for epoch in range(1, EXTERNAL_PRETRAIN_EPOCHS + 1):
            train_loss = common.train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                english_sp.pad_id(),
            )

            run.log(
                {
                    "pretrain/epoch": epoch,
                    "pretrain/train_loss": train_loss,
                }
            )

            save_pretrained_checkpoint(
                model=model,
                epoch=epoch,
                train_loss=train_loss,
                problem97_checkpoint=problem97_checkpoint,
                external_train_sentences=len(external_dataset),
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
            )

            print(
                f"JParaCrawl epoch {epoch}/{EXTERNAL_PRETRAIN_EPOCHS} | "
                f"train loss={train_loss:.4f}"
            )
            print("事前学習checkpointを保存:", PRETRAINED_MODEL_PATH)

    finally:
        del train_loader, criterion, optimizer
        cleanup_gpu()


# ============================================================
# 9. KFTTによるファインチューニング
# ============================================================


def finetune_with_kftt(
    model: nn.Module,
    common: ModuleType,
    problem93: ModuleType,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    problem97_checkpoint: dict[str, object],
    external_train_sentences: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    device: torch.device,
    run: wandb.sdk.wandb_run.Run,
) -> list[dict[str, int | float]]:
    """事前学習済みモデルをKFTT trainでファインチューニングする。"""
    print("\n" + "=" * 70)
    print("KFTTによるファインチューニング")
    print(f"KFTT train文対数: {len(train_dataset):,}")
    print(f"最大epoch: {KFTT_FINETUNE_EPOCHS}")
    print("=" * 70)

    train_loader = common.create_data_loader(
        train_dataset,
        batch_size,
        True,
        japanese_sp.pad_id(),
        english_sp.pad_id(),
    )
    dev_loader = common.create_data_loader(
        dev_dataset,
        EVAL_BATCH_SIZE,
        False,
        japanese_sp.pad_id(),
        english_sp.pad_id(),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=english_sp.pad_id())

    # 事前学習とファインチューニングではOptimizerを作り直す。
    # これにより、JParaCrawlで蓄積したAdamの内部状態を引き継がず、
    # KFTTへの適応を新しい最適化段階として始める。
    optimizer = create_optimizer(model, optimizer_name, learning_rate)

    history: list[dict[str, int | float]] = []
    best_dev_bleu = -float("inf")
    epochs_without_improvement = 0

    max_epochs = (
        DEBUG_FINETUNE_EPOCHS
        if DEBUG
        else KFTT_FINETUNE_EPOCHS
    )

    try:
        for epoch in range(1, max_epochs + 1):
            train_loss = common.train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                english_sp.pad_id(),
            )
            dev_loss = common.evaluate_loss(
                model,
                dev_loader,
                criterion,
                device,
                english_sp.pad_id(),
                description="98 KFTT dev loss",
            )
            dev_bleu = common.calculate_bleu(
                model,
                dev_loader,
                english_sp,
                problem93.detokenize_english,
                device,
                description="98 KFTT dev BLEU",
            )

            epoch_result: dict[str, int | float] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "dev_bleu": dev_bleu,
            }
            history.append(epoch_result)
            save_json(HISTORY_PATH, history)

            run.log(
                {
                    "finetune/epoch": epoch,
                    "finetune/train_loss": train_loss,
                    "finetune/dev_loss": dev_loss,
                    "finetune/dev_bleu": dev_bleu,
                }
            )

            print(
                f"KFTT epoch {epoch}/{max_epochs} | "
                f"train loss={train_loss:.4f} | "
                f"dev loss={dev_loss:.4f} | "
                f"dev BLEU={dev_bleu:.2f}"
            )

            if dev_bleu > best_dev_bleu:
                best_dev_bleu = dev_bleu
                epochs_without_improvement = 0

                save_best_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    dev_loss=dev_loss,
                    dev_bleu=dev_bleu,
                    problem97_checkpoint=problem97_checkpoint,
                    external_train_sentences=external_train_sentences,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer_name,
                )
                print("98番の最良モデルを保存:", BEST_MODEL_PATH)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    "dev BLEUが改善しない状態が"
                    f"{EARLY_STOPPING_PATIENCE}エポック続いたため、"
                    "ファインチューニングを早期終了します。"
                )
                break

        run.summary["best_finetune_dev_bleu"] = best_dev_bleu
        return history

    finally:
        del train_loader, dev_loader, criterion, optimizer
        cleanup_gpu()


# ============================================================
# 10. KFTT test BLEUの計算
# ============================================================


def load_best_98_model(
    problem91: ModuleType,
    problem97_checkpoint: dict[str, object],
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    """98番の最良checkpointからTransformerを復元する。"""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"98番の最良モデルがありません: {BEST_MODEL_PATH}")

    model = create_model_from_problem97_config(
        problem91,
        problem97_checkpoint,
        device,
    )
    checkpoint = load_model_state(model, BEST_MODEL_PATH)
    model.eval()
    return model, checkpoint


def evaluate_test_bleu(
    common: ModuleType,
    problem91: ModuleType,
    problem93: ModuleType,
    problem97_checkpoint: dict[str, object],
    test_dataset: Dataset,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    """保存済み最良モデルをKFTT test全件で1回だけ評価する。"""
    model, checkpoint = load_best_98_model(
        problem91,
        problem97_checkpoint,
        device,
    )

    test_loader = common.create_data_loader(
        test_dataset,
        EVAL_BATCH_SIZE,
        False,
        japanese_sp.pad_id(),
        english_sp.pad_id(),
    )

    test_bleu = common.calculate_bleu(
        model,
        test_loader,
        english_sp,
        problem93.detokenize_english,
        device,
        description="98 KFTT test BLEU",
        prediction_path=TEST_PREDICTION_PATH,
        reference_path=TEST_REFERENCE_PATH,
    )

    result = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_dev_bleu": float(checkpoint["dev_bleu"]),
        "test_bleu": test_bleu,
    }

    del model, test_loader
    cleanup_gpu()
    return test_bleu, result


# ============================================================
# 11. 97番との比較結果を保存・可視化する
# ============================================================


def save_test_comparison_plot(
    baseline_test_bleu: float,
    adapted_test_bleu: float,
) -> None:
    """97番と98番のKFTT test BLEUを棒グラフにする。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    labels = [
        "97: KFTT only",
        "98: JParaCrawl -> KFTT",
    ]
    scores = [baseline_test_bleu, adapted_test_bleu]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, scores)
    plt.ylabel("KFTT test BLEU")
    plt.title("Problem 98: effect of external parallel data")

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score,
            f"{score:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(TEST_COMPARISON_PLOT_PATH, dpi=150)
    plt.close()


def save_dev_curve_plot(
    problem98_history: list[dict[str, int | float]],
) -> None:
    """同じKFTTエポック数で97番と98番のdev BLEU推移を比較する。"""
    problem97_history = load_json(PROBLEM97_HISTORY_PATH)
    if not isinstance(problem97_history, list):
        raise TypeError("97_final_history.jsonがリストではありません。")

    problem97_epochs = [int(row["epoch"]) for row in problem97_history]
    problem97_bleu = [float(row["dev_bleu"]) for row in problem97_history]

    problem98_epochs = [int(row["epoch"]) for row in problem98_history]
    problem98_bleu = [float(row["dev_bleu"]) for row in problem98_history]

    plt.figure(figsize=(8, 5))
    plt.plot(
        problem97_epochs,
        problem97_bleu,
        marker="o",
        label="97: KFTT only",
    )
    plt.plot(
        problem98_epochs,
        problem98_bleu,
        marker="o",
        label="98: JParaCrawl -> KFTT",
    )
    plt.xlabel("KFTT training epoch")
    plt.ylabel("KFTT development BLEU")
    plt.title("Development BLEU during KFTT training")
    plt.xticks(sorted(set(problem97_epochs + problem98_epochs)))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(DEV_CURVE_PLOT_PATH, dpi=150)
    plt.close()


def save_final_result(
    baseline_test_bleu: float,
    adapted_test_bleu: float,
    test_information: dict[str, object],
    external_train_sentences: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
) -> dict[str, object]:
    """98番の設定・BLEU・97番との差をJSONへ保存する。"""
    improvement = adapted_test_bleu - baseline_test_bleu

    result: dict[str, object] = {
        "external_corpus": "JParaCrawl v3.0",
        "external_train_sentences": external_train_sentences,
        "external_pretrain_epochs": EXTERNAL_PRETRAIN_EPOCHS,
        "kftt_max_finetune_epochs": KFTT_FINETUNE_EPOCHS,
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer_name": optimizer_name,
        },
        "problem97_baseline_test_bleu": baseline_test_bleu,
        "problem98_test_bleu": adapted_test_bleu,
        "improvement": improvement,
        "performance_improved": improvement > 0.0,
        **test_information,
    }
    save_json(TEST_RESULT_PATH, result)
    return result


# ============================================================
# 12. main
# ============================================================


def main() -> None:
    check_required_files()
    set_random_seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("98番はCUDA対応GPU上で実行してください。")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    # これまでのクラス・評価関数を読み込む。
    problem91 = load_module("problem91_for_98", PROBLEM91_PATH)
    problem93 = load_module("problem93_for_98", PROBLEM93_PATH)
    common = load_module("problem95_96_for_98", PROBLEM95_96_PATH)

    # 97番から、比較基準と最良ハイパーパラメータを得る。
    problem97_checkpoint, baseline_test_bleu = (
        load_problem97_information()
    )
    batch_size, learning_rate, optimizer_name = (
        get_best_hyperparameters(problem97_checkpoint)
    )

    print("\n97番の基準")
    print(f"KFTT test BLEU: {baseline_test_bleu:.2f}")
    print(f"batch size: {batch_size}")
    print(f"learning rate: {learning_rate}")
    print(f"optimizer: {optimizer_name}")

    # 95・96番で作ったSentencePieceモデルとKFTTサブワードを再利用する。
    common.check_input_files()
    japanese_sp, english_sp = common.prepare_subword_data()

    # 97番checkpointと現在のSentencePieceの語彙数が一致するか確認する。
    if japanese_sp.get_piece_size() != int(
        problem97_checkpoint["japanese_vocab_size"]
    ):
        raise ValueError(
            "97番と現在の日本語SentencePiece語彙数が一致しません。"
        )
    if english_sp.get_piece_size() != int(
        problem97_checkpoint["english_vocab_size"]
    ):
        raise ValueError(
            "97番と現在の英語SentencePiece語彙数が一致しません。"
        )

    # 事前学習checkpointを再利用しない場合だけ、
    # JParaCrawlの前処理とDataset作成が必要になる。
    external_metadata: dict[str, int | float | str]
    external_dataset: Dataset | None = None

    if pretrained_checkpoint_is_compatible():
        print("\n現在の設定と互換性があるJParaCrawl事前学習checkpointがあります。")
        print(PRETRAINED_MODEL_PATH)

        pretrained_checkpoint = torch.load(
            PRETRAINED_MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
        external_train_sentences = int(
            pretrained_checkpoint["external_train_sentences"]
        )

        if EXTERNAL_METADATA_PATH.exists():
            loaded_metadata = load_json(EXTERNAL_METADATA_PATH)
            external_metadata = (
                loaded_metadata
                if isinstance(loaded_metadata, dict)
                else {}
            )
        else:
            external_metadata = {
                "corpus": "JParaCrawl v3.0",
                "written_pairs": external_train_sentences,
            }
    else:
        external_metadata = prepare_external_subword_data(
            japanese_sp,
            english_sp,
        )
        external_dataset = create_external_dataset(
            common,
            japanese_sp,
            english_sp,
        )
        external_train_sentences = len(external_dataset)

    train_dataset, dev_dataset, test_dataset = create_kftt_datasets(
        common,
        japanese_sp,
        english_sp,
    )

    # 97番と同じ構造だが、新しいランダム初期値を持つモデルを作る。
    # 97_best_model.ptの重みは読み込まない。
    set_random_seed(RANDOM_SEED)
    model = create_model_from_problem97_config(
        problem91,
        problem97_checkpoint,
        device,
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        mode=WANDB_MODE,
        config={
            "external_corpus": "JParaCrawl v3.0",
            "external_train_sentences": external_train_sentences,
            "external_pretrain_epochs": EXTERNAL_PRETRAIN_EPOCHS,
            "kftt_finetune_epochs": (
                DEBUG_FINETUNE_EPOCHS
                if DEBUG
                else KFTT_FINETUNE_EPOCHS
            ),
            "batch_size": batch_size,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "baseline_97_test_bleu": baseline_test_bleu,
            "debug": DEBUG,
            "model_config": problem97_checkpoint["model_config"],
            "external_metadata": external_metadata,
        },
    )

    try:
        # 1. 外部対訳データで一般的な日英翻訳能力を事前学習する。
        if external_dataset is None and not PRETRAINED_MODEL_PATH.exists():
            raise RuntimeError(
                "外部Datasetも事前学習checkpointもありません。"
            )

        if external_dataset is None:
            # checkpoint再利用時にはDatasetを作らなくてもよい。
            load_model_state(model, PRETRAINED_MODEL_PATH)
            print("JParaCrawl事前学習済み重みを読み込みました。")
        else:
            pretrain_with_jparacrawl(
                model=model,
                common=common,
                external_dataset=external_dataset,
                japanese_sp=japanese_sp,
                english_sp=english_sp,
                problem97_checkpoint=problem97_checkpoint,
                batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                device=device,
                run=run,
            )

        # 2. 同じモデルをKFTTドメインへ適応させる。
        history = finetune_with_kftt(
            model=model,
            common=common,
            problem93=problem93,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            japanese_sp=japanese_sp,
            english_sp=english_sp,
            problem97_checkpoint=problem97_checkpoint,
            external_train_sentences=external_train_sentences,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            device=device,
            run=run,
        )

        # 3. dev BLEUで選んだ1モデルだけをKFTT testで評価する。
        adapted_test_bleu, test_information = evaluate_test_bleu(
            common=common,
            problem91=problem91,
            problem93=problem93,
            problem97_checkpoint=problem97_checkpoint,
            test_dataset=test_dataset,
            japanese_sp=japanese_sp,
            english_sp=english_sp,
            device=device,
        )

        final_result = save_final_result(
            baseline_test_bleu=baseline_test_bleu,
            adapted_test_bleu=adapted_test_bleu,
            test_information=test_information,
            external_train_sentences=external_train_sentences,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
        )

        save_test_comparison_plot(
            baseline_test_bleu,
            adapted_test_bleu,
        )
        save_dev_curve_plot(history)

        improvement = float(final_result["improvement"])
        run.summary["problem98_test_bleu"] = adapted_test_bleu
        run.summary["improvement_over_problem97"] = improvement
        run.summary["performance_improved"] = improvement > 0.0

    finally:
        run.finish()
        del model
        cleanup_gpu()

    print("\n" + "=" * 70)
    print("98番の最終結果")
    print("=" * 70)
    print("学習方法: JParaCrawl事前学習 -> KFTTファインチューニング")
    print(f"外部学習文対数: {external_train_sentences:,}")
    print(f"97番 KFTT-only test BLEU: {baseline_test_bleu:.2f}")
    print(f"98番 domain-adapted test BLEU: {adapted_test_bleu:.2f}")
    print(f"差分: {improvement:+.2f}")

    if improvement > 0.0:
        print("KFTT test BLEUの性能向上を確認できました。")
    else:
        print(
            "今回はKFTT test BLEUの向上を確認できませんでした。\n"
            "ただし、外部データを活用して性能向上を試みたという"
            "98番の実験結果として、そのまま正確に報告できます。"
        )

    print("\n保存先")
    print("JParaCrawl metadata:", EXTERNAL_METADATA_PATH)
    print("JParaCrawl preview:", EXTERNAL_PREVIEW_PATH)
    print("pretrained model:", PRETRAINED_MODEL_PATH)
    print("best adapted model:", BEST_MODEL_PATH)
    print("history:", HISTORY_PATH)
    print("test result:", TEST_RESULT_PATH)
    print("test predictions:", TEST_PREDICTION_PATH)
    print("test references:", TEST_REFERENCE_PATH)
    print("test BLEU comparison plot:", TEST_COMPARISON_PLOT_PATH)
    print("dev BLEU curve plot:", DEV_CURVE_PLOT_PATH)


if __name__ == "__main__":
    main()
