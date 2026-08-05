"""97: ハイパーパラメータの調整。

95・96番で作成したSentencePieceモデル、サブワード化済みKFTT、
Transformer学習・BLEU計算の関数を再利用し、次を比較する。

- バッチサイズ: 8, 16, 32
- 学習率: 1e-3, 1e-4, 1e-5, 1e-6
- Optimizer: Adam, AdamW, RAdam

全組合せを試すと計算時間が非常に長くなるため、このコードでは
「基準設定から1項目ずつ変更する8実験」で候補を絞る。
候補探索は固定した訓練データの一部で行い、最良設定だけを
訓練データ全体で改めて学習する。

最後に、最良モデルのKFTT test BLEUを計算し、BLEU 10以上か確認する。
"""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import sentencepiece as spm
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# 1. ファイルの場所
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# 91.pyからTransformerクラスとモデル構成を読み込む。
PROBLEM91_PATH = BASE_DIR / "91.py"

# 93.pyから英文を読みやすい形へ戻す関数を読み込む。
PROBLEM93_PATH = BASE_DIR / "93.py"

# 95・96.pyからSentencePiece、Dataset、学習、BLEU計算を再利用する。
PROBLEM95_96_PATH = BASE_DIR / "95_96.py"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

# 97番で最終的に保存する最良モデル。
BEST_MODEL_PATH = MODEL_DIR / "97_best_model.pt"

# ハイパーパラメータ探索の全結果。
TUNING_CSV_PATH = RESULT_DIR / "97_tuning_results.csv"
TUNING_JSON_PATH = RESULT_DIR / "97_tuning_results.json"
TUNING_PLOT_PATH = RESULT_DIR / "97_tuning_results.png"

# 最良設定を使った最終学習のエポックごとの結果。
FINAL_HISTORY_PATH = RESULT_DIR / "97_final_history.json"

# 最良モデルをtestデータで評価した結果。
TEST_BLEU_PATH = RESULT_DIR / "97_test_bleu.json"
TEST_PREDICTION_PATH = RESULT_DIR / "97_test_predictions.en"
TEST_REFERENCE_PATH = RESULT_DIR / "97_test_references.en"


# ============================================================
# 2. 実験設定
# ============================================================

RANDOM_SEED = 42
MAX_SENTENCE_LENGTH = 256

# -------------------- 候補探索 --------------------

# 8設定すべてを44万文で学習すると非常に時間がかかるため、
# 候補探索では固定した10万文を使う。
# Noneにすると、候補探索でも訓練データ全体を使う。
TUNING_TRAIN_SENTENCES: int | None = 100_000

# 候補探索は各設定1エポックだけ行う。
# ここでは厳密な最終評価ではなく、有望な設定を絞ることが目的である。
TUNING_EPOCHS = 1

# -------------------- 最終学習 --------------------

# 候補探索で最良だった設定を、訓練データ全体で学習する。
# 97番ではBLEU 10以上が必要なため、95・96番より少し多い5エポックとする。
FINAL_EPOCHS = 5

# dev BLEUが連続して改善しない場合は、無駄な学習を早めに止める。
EARLY_STOPPING_PATIENCE = 2

# dev/test評価は学習時のバッチサイズとは独立してよいので固定する。
# GPUメモリ不足になる場合は16へ下げる。
EVAL_BATCH_SIZE = 32

GRADIENT_CLIP_NORM = 1.0

# GPUサーバからW&Bへ接続できない場合は"offline"へ変更できる。
WANDB_MODE = "online"
WANDB_PROJECT = "nlp100-machine-translation"
WANDB_TUNING_GROUP = "97-hyperparameter-tuning"


@dataclass(frozen=True)
class Experiment:
    """1回の候補探索で使用するハイパーパラメータ。"""

    name: str
    batch_size: int
    learning_rate: float
    optimizer_name: str


# 全組合せは3×4×3=36実験になり、計算時間が非常に長い。
# そこで、Adam・batch size 16・learning rate 1e-4を基準にし、
# 1項目ずつ変えた8設定を比較する。
EXPERIMENTS = [
    Experiment("baseline-adam-bs16-lr1e-4", 16, 1e-4, "Adam"),
    Experiment("batch-bs8", 8, 1e-4, "Adam"),
    Experiment("batch-bs32", 32, 1e-4, "Adam"),
    Experiment("lr-1e-3", 16, 1e-3, "Adam"),
    Experiment("lr-1e-5", 16, 1e-5, "Adam"),
    Experiment("lr-1e-6", 16, 1e-6, "Adam"),
    Experiment("optimizer-adamw", 16, 1e-4, "AdamW"),
    Experiment("optimizer-radam", 16, 1e-4, "RAdam"),
]


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

    # @dataclassなどは、クラスが定義されたモジュールを
    # sys.modulesから参照するため、実行前に登録しておく。
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        # 読み込みに失敗した場合は、
        # 不完全なモジュールをsys.modulesに残さない。
        sys.modules.pop(name, None)
        raise

    return module


def check_required_files() -> None:
    """97番が再利用する91・93・95_96.pyの存在を確認する。"""
    required_paths = [
        PROBLEM91_PATH,
        PROBLEM93_PATH,
        PROBLEM95_96_PATH,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"必要なファイルがありません:\n{missing_text}")


def model_config(problem91: ModuleType) -> dict[str, int | float]:
    """91.pyと同じTransformer構成を辞書にまとめる。"""
    return {
        "d_model": problem91.D_MODEL,
        "nhead": problem91.NHEAD,
        "num_encoder_layers": problem91.NUM_ENCODER_LAYERS,
        "num_decoder_layers": problem91.NUM_DECODER_LAYERS,
        "dim_feedforward": problem91.DIM_FEEDFORWARD,
        "dropout": problem91.DROPOUT,
        "max_position_length": problem91.MAX_POSITION_LENGTH,
    }


def create_model(
    problem91: ModuleType,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    device: torch.device,
) -> nn.Module:
    """毎実験で同じ構造の新しいTransformerを作る。"""
    model = problem91.TranslationTransformer(
        japanese_vocab_size=japanese_sp.get_piece_size(),
        english_vocab_size=english_sp.get_piece_size(),
        japanese_pad_id=japanese_sp.pad_id(),
        english_pad_id=english_sp.pad_id(),
        **model_config(problem91),
    )
    return model.to(device)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """名前に応じてAdam系Optimizerを作る。"""
    if optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    if optimizer_name == "AdamW":
        # AdamWはweight decayを勾配更新から分離して適用する。
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

    if optimizer_name == "RAdam":
        # RAdamは学習初期の更新量を自動的に調整するAdam系Optimizer。
        return torch.optim.RAdam(
            model.parameters(),
            lr=learning_rate,
        )

    raise ValueError(f"未対応のOptimizerです: {optimizer_name}")


def create_fixed_subset(
    dataset: Dataset,
    sample_size: int | None,
    seed: int,
) -> Dataset:
    """毎実験で同じ文を使う、固定された訓練サブセットを作る。"""
    if sample_size is None or sample_size >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:sample_size].tolist()

    return Subset(dataset, indices)


def is_recoverable_training_error(error: RuntimeError) -> bool:
    """設定変更で起こり得るOOMまたはNaNかを判定する。"""
    message = str(error).lower()
    return (
        "out of memory" in message
        or "nan" in message
        or "無限大" in message
    )


def cleanup_gpu() -> None:
    """次の実験へ移る前に、不要なGPUメモリを解放する。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 4. 結果の保存とグラフ化
# ============================================================


def save_tuning_results(results: list[dict[str, object]]) -> None:
    """候補探索の途中結果をCSV・JSON・グラフへ保存する。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    TUNING_JSON_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "name",
        "batch_size",
        "learning_rate",
        "optimizer",
        "epochs",
        "train_sentences",
        "best_epoch",
        "train_loss",
        "dev_loss",
        "dev_bleu",
        "status",
        "error",
    ]
    with TUNING_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 正常終了した実験だけをBLEUの比較グラフに載せる。
    successful_results = [
        result
        for result in results
        if result["status"] == "ok"
    ]
    if not successful_results:
        return

    names = [str(result["name"]) for result in successful_results]
    scores = [float(result["dev_bleu"]) for result in successful_results]

    plt.figure(figsize=(10, 6))
    plt.barh(names, scores)
    plt.xlabel("Best development BLEU")
    plt.ylabel("Hyperparameter setting")
    plt.title("Problem 97: hyperparameter tuning")
    plt.tight_layout()
    plt.savefig(TUNING_PLOT_PATH, dpi=150)
    plt.close()


def save_final_history(history: list[dict[str, int | float]]) -> None:
    """最終学習のエポックごとの結果をJSONへ保存する。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# 5. 候補探索
# ============================================================


def run_one_tuning_experiment(
    experiment: Experiment,
    common: ModuleType,
    problem91: ModuleType,
    problem93: ModuleType,
    tuning_dataset: Dataset,
    dev_dataset: Dataset,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    device: torch.device,
) -> dict[str, object]:
    """1つの設定でモデルを最初から学習し、dev BLEUを返す。"""
    print("\n" + "=" * 70)
    print(f"候補探索: {experiment.name}")
    print(asdict(experiment))
    print("=" * 70)

    # どの設定も同じ初期重み・同じデータ順から開始し、比較を公平にする。
    set_random_seed(RANDOM_SEED)

    train_loader = common.create_data_loader(
        tuning_dataset,
        experiment.batch_size,
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

    model = create_model(problem91, japanese_sp, english_sp, device)
    criterion = nn.CrossEntropyLoss(ignore_index=english_sp.pad_id())
    optimizer = create_optimizer(
        model,
        experiment.optimizer_name,
        experiment.learning_rate,
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"97-tuning-{experiment.name}",
        group=WANDB_TUNING_GROUP,
        mode=WANDB_MODE,
        config={
            **asdict(experiment),
            "stage": "tuning",
            "epochs": TUNING_EPOCHS,
            "train_sentences": len(tuning_dataset),
            "eval_batch_size": EVAL_BATCH_SIZE,
            **model_config(problem91),
        },
    )

    best_epoch = 0
    best_dev_bleu = -float("inf")
    best_train_loss = float("inf")
    best_dev_loss = float("inf")

    try:
        for epoch in range(1, TUNING_EPOCHS + 1):
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
                description=f"{experiment.name} dev loss",
            )
            dev_bleu = common.calculate_bleu(
                model,
                dev_loader,
                english_sp,
                problem93.detokenize_english,
                device,
                description=f"{experiment.name} dev BLEU",
            )

            run.log(
                {
                    "train/loss": train_loss,
                    "dev/loss": dev_loss,
                    "dev/bleu": dev_bleu,
                },
                step=epoch,
            )

            print(
                f"epoch {epoch}/{TUNING_EPOCHS} | "
                f"train loss={train_loss:.4f} | "
                f"dev loss={dev_loss:.4f} | "
                f"dev BLEU={dev_bleu:.2f}"
            )

            if dev_bleu > best_dev_bleu:
                best_epoch = epoch
                best_dev_bleu = dev_bleu
                best_train_loss = train_loss
                best_dev_loss = dev_loss

        run.summary["best_epoch"] = best_epoch
        run.summary["best_dev_bleu"] = best_dev_bleu

        return {
            "name": experiment.name,
            "batch_size": experiment.batch_size,
            "learning_rate": experiment.learning_rate,
            "optimizer": experiment.optimizer_name,
            "epochs": TUNING_EPOCHS,
            "train_sentences": len(tuning_dataset),
            "best_epoch": best_epoch,
            "train_loss": best_train_loss,
            "dev_loss": best_dev_loss,
            "dev_bleu": best_dev_bleu,
            "status": "ok",
            "error": "",
        }

    except RuntimeError as error:
        if not is_recoverable_training_error(error):
            raise

        # 大きいバッチサイズのOOMや大きい学習率のNaNは、
        # 「その設定では学習できなかった」という探索結果として記録する。
        error_text = str(error)
        print(f"この設定は失敗しました: {error_text}")
        run.summary["status"] = "failed"
        run.summary["error"] = error_text

        return {
            "name": experiment.name,
            "batch_size": experiment.batch_size,
            "learning_rate": experiment.learning_rate,
            "optimizer": experiment.optimizer_name,
            "epochs": TUNING_EPOCHS,
            "train_sentences": len(tuning_dataset),
            "best_epoch": 0,
            "train_loss": "",
            "dev_loss": "",
            "dev_bleu": -1.0,
            "status": "failed",
            "error": error_text,
        }

    finally:
        run.finish()
        del model, optimizer, criterion, train_loader, dev_loader
        cleanup_gpu()


def run_hyperparameter_tuning(
    common: ModuleType,
    problem91: ModuleType,
    problem93: ModuleType,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    device: torch.device,
) -> tuple[Experiment, list[dict[str, object]]]:
    """全候補を比較し、dev BLEUが最大の設定を返す。"""
    tuning_dataset = create_fixed_subset(
        train_dataset,
        TUNING_TRAIN_SENTENCES,
        RANDOM_SEED,
    )
    print(f"候補探索に使う訓練文対: {len(tuning_dataset):,}")

    results: list[dict[str, object]] = []

    for experiment in EXPERIMENTS:
        result = run_one_tuning_experiment(
            experiment=experiment,
            common=common,
            problem91=problem91,
            problem93=problem93,
            tuning_dataset=tuning_dataset,
            dev_dataset=dev_dataset,
            japanese_sp=japanese_sp,
            english_sp=english_sp,
            device=device,
        )
        results.append(result)

        # 長時間実験の途中で停止しても、それまでの結果が残るよう毎回保存する。
        save_tuning_results(results)

    successful_results = [
        result
        for result in results
        if result["status"] == "ok"
    ]
    if not successful_results:
        raise RuntimeError("正常に終了したハイパーパラメータ設定がありません。")

    best_result = max(
        successful_results,
        key=lambda result: float(result["dev_bleu"]),
    )

    best_experiment = next(
        experiment
        for experiment in EXPERIMENTS
        if experiment.name == best_result["name"]
    )

    print("\n候補探索の最良設定")
    print(f"name: {best_experiment.name}")
    print(f"batch size: {best_experiment.batch_size}")
    print(f"learning rate: {best_experiment.learning_rate}")
    print(f"optimizer: {best_experiment.optimizer_name}")
    print(f"dev BLEU: {float(best_result['dev_bleu']):.2f}")

    return best_experiment, results


# ============================================================
# 6. 最良設定による訓練データ全体での学習
# ============================================================


def save_best_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    experiment: Experiment,
    epoch: int,
    train_loss: float,
    dev_loss: float,
    dev_bleu: float,
    problem91: ModuleType,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
) -> None:
    """最良dev BLEUを更新したモデルを保存する。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "dev_bleu": dev_bleu,
        "hyperparameters": asdict(experiment),
        "model_config": model_config(problem91),
        "japanese_vocab_size": japanese_sp.get_piece_size(),
        "english_vocab_size": english_sp.get_piece_size(),
        "special_token_ids": {
            "japanese_pad_id": japanese_sp.pad_id(),
            "english_pad_id": english_sp.pad_id(),
        },
    }
    torch.save(checkpoint, BEST_MODEL_PATH)


def train_final_model(
    best_experiment: Experiment,
    common: ModuleType,
    problem91: ModuleType,
    problem93: ModuleType,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    japanese_sp: spm.SentencePieceProcessor,
    english_sp: spm.SentencePieceProcessor,
    device: torch.device,
) -> None:
    """最良設定を使い、訓練データ全体でモデルを最初から学習する。"""
    print("\n" + "=" * 70)
    print("最良設定による最終学習")
    print(asdict(best_experiment))
    print(f"訓練文対: {len(train_dataset):,}")
    print("=" * 70)

    set_random_seed(RANDOM_SEED)

    train_loader = common.create_data_loader(
        train_dataset,
        best_experiment.batch_size,
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

    model = create_model(problem91, japanese_sp, english_sp, device)
    criterion = nn.CrossEntropyLoss(ignore_index=english_sp.pad_id())
    optimizer = create_optimizer(
        model,
        best_experiment.optimizer_name,
        best_experiment.learning_rate,
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name="97-final-best-model",
        group=WANDB_TUNING_GROUP,
        mode=WANDB_MODE,
        config={
            **asdict(best_experiment),
            "stage": "final",
            "epochs": FINAL_EPOCHS,
            "train_sentences": len(train_dataset),
            "eval_batch_size": EVAL_BATCH_SIZE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            **model_config(problem91),
        },
    )

    history: list[dict[str, int | float]] = []
    best_dev_bleu = -float("inf")
    epochs_without_improvement = 0

    try:
        for epoch in range(1, FINAL_EPOCHS + 1):
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
                description="final dev loss",
            )
            dev_bleu = common.calculate_bleu(
                model,
                dev_loader,
                english_sp,
                problem93.detokenize_english,
                device,
                description="final dev BLEU",
            )

            epoch_result: dict[str, int | float] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "dev_bleu": dev_bleu,
            }
            history.append(epoch_result)
            save_final_history(history)

            run.log(
                {
                    "train/loss": train_loss,
                    "dev/loss": dev_loss,
                    "dev/bleu": dev_bleu,
                },
                step=epoch,
            )

            print(
                f"epoch {epoch}/{FINAL_EPOCHS} | "
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
                    experiment=best_experiment,
                    epoch=epoch,
                    train_loss=train_loss,
                    dev_loss=dev_loss,
                    dev_bleu=dev_bleu,
                    problem91=problem91,
                    japanese_sp=japanese_sp,
                    english_sp=english_sp,
                )
                print(f"最良モデルを保存: {BEST_MODEL_PATH}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    "dev BLEUが改善しない状態が"
                    f"{EARLY_STOPPING_PATIENCE}エポック続いたため、"
                    "最終学習を早期終了します。"
                )
                break

        run.summary["best_dev_bleu"] = best_dev_bleu

    finally:
        run.finish()
        del model, optimizer, criterion, train_loader, dev_loader
        cleanup_gpu()


# ============================================================
# 7. 保存済み最良モデルの読み込みとtest評価
# ============================================================


def load_best_model(
    problem91: ModuleType,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    """97番の最良checkpointからTransformerを復元する。"""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"最良モデルがありません: {BEST_MODEL_PATH}")

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

    return model, checkpoint


def evaluate_test_bleu(
    common: ModuleType,
    problem91: ModuleType,
    problem93: ModuleType,
    test_dataset: Dataset,
    english_sp: spm.SentencePieceProcessor,
    japanese_sp: spm.SentencePieceProcessor,
    device: torch.device,
) -> tuple[float, dict[str, object]]:
    """最良モデルをKFTT testデータ全体で評価する。"""
    model, checkpoint = load_best_model(problem91, device)

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
        description="97 test BLEU",
        prediction_path=TEST_PREDICTION_PATH,
        reference_path=TEST_REFERENCE_PATH,
    )

    result = {
        "best_epoch": checkpoint["epoch"],
        "best_dev_bleu": checkpoint["dev_bleu"],
        "hyperparameters": checkpoint["hyperparameters"],
        "test_bleu": test_bleu,
        "bleu_at_least_10": test_bleu >= 10.0,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TEST_BLEU_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model, test_loader
    cleanup_gpu()
    return test_bleu, result


# ============================================================
# 8. main
# ============================================================


def main() -> None:
    check_required_files()
    set_random_seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("97番はCUDA対応GPU上で実行してください。")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    problem91 = load_module("problem91_for_97", PROBLEM91_PATH)
    problem93 = load_module("problem93_for_97", PROBLEM93_PATH)
    common = load_module("problem95_96_for_97", PROBLEM95_96_PATH)

    # 95・96番で作ったSentencePieceモデルとサブワードデータを再利用する。
    # ファイルがなければ、95_96.pyの関数が作成する。
    common.check_input_files()
    japanese_sp, english_sp = common.prepare_subword_data()

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

    # 1. 必須3項目を変えた候補を比較する。
    best_experiment, _ = run_hyperparameter_tuning(
        common=common,
        problem91=problem91,
        problem93=problem93,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        japanese_sp=japanese_sp,
        english_sp=english_sp,
        device=device,
    )

    # 2. 最良設定だけを訓練データ全体で改めて学習する。
    train_final_model(
        best_experiment=best_experiment,
        common=common,
        problem91=problem91,
        problem93=problem93,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        japanese_sp=japanese_sp,
        english_sp=english_sp,
        device=device,
    )

    # 3. 保存した最良モデルをtestデータで1回だけ最終評価する。
    test_bleu, final_result = evaluate_test_bleu(
        common=common,
        problem91=problem91,
        problem93=problem93,
        test_dataset=test_dataset,
        english_sp=english_sp,
        japanese_sp=japanese_sp,
        device=device,
    )

    print("\n" + "=" * 70)
    print("97番の最終結果")
    print("=" * 70)
    print("best hyperparameters:", final_result["hyperparameters"])
    print(f"best epoch: {final_result['best_epoch']}")
    print(f"best dev BLEU: {float(final_result['best_dev_bleu']):.2f}")
    print(f"test BLEU: {test_bleu:.2f}")

    if test_bleu >= 10.0:
        print("BLEU 10以上を確認できました。")
    else:
        print(
            "BLEUが10未満です。FINAL_EPOCHS、探索する設定、"
            "モデルサイズなどを追加調整してください。"
        )

    print("\n保存先")
    print("best model:", BEST_MODEL_PATH)
    print("tuning CSV:", TUNING_CSV_PATH)
    print("tuning JSON:", TUNING_JSON_PATH)
    print("tuning plot:", TUNING_PLOT_PATH)
    print("final history:", FINAL_HISTORY_PATH)
    print("test BLEU:", TEST_BLEU_PATH)
    print("test predictions:", TEST_PREDICTION_PATH)
    print("test references:", TEST_REFERENCE_PATH)


if __name__ == "__main__":
    main()

#出力
# ======================================================================
# 97番の最終結果
# ======================================================================
# best hyperparameters: {'name': 'batch-bs8', 'batch_size': 8, 'learning_rate': 0.0001, 'optimizer_name': 'Adam'}
# best epoch: 5
# best dev BLEU: 17.02
# test BLEU: 19.11
# BLEU 10以上を確認できました。