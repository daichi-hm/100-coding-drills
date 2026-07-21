from pathlib import Path
import csv
import gc
import random

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


# ============================================================
# 設定
# ============================================================

MODEL_NAME = "openai-community/gpt2-medium"

# True:
#   98番でファインチューニングしたGPT-2-mediumを、
#   DPOの開始モデルとして使う
#
# False:
#   元のGPT-2-mediumから直接DPOを始める
START_FROM_98_MODEL = True

# prompt + chosen/rejected の最大トークン数
MAX_LENGTH = 128

# DPOでは、同じpromptに対してchosenとrejectedの両方を計算し、
# さらにreference modelとの比較も行うため、98番より重くなる。
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

# 今回は勾配蓄積を行わず、毎バッチ更新する
GRADIENT_ACCUMULATION_STEPS = 1

EPOCHS = 1

# DPOの全パラメータ学習では、通常のSFTより小さめの学習率を使う
LEARNING_RATE = 1e-6

# DPOで、元のreference modelからどの程度離れにくくするかを調整する
BETA = 0.1

SEED = 42

# 最初の動作確認ではTrueにする
DEBUG = True

# DEBUG=Trueのときに使用する件数
DEBUG_TRAIN_SIZE = 512
DEBUG_DEV_SIZE = 100


# ============================================================
# 乱数の固定
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SST-2の読み込み
# ============================================================

def load_sst2(path: Path) -> tuple[list[str], list[int]]:
    """
    SST-2のTSVファイルから文とラベルを読み込む。

    label:
        0 = negative
        1 = positive
    """

    texts: list[str] = []
    labels: list[int] = []

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            texts.append(row["sentence"])
            labels.append(int(row["label"]))

    return texts, labels


# ============================================================
# 96番・98番と同じプロンプト
# ============================================================

def build_prompt(sentence: str) -> str:
    """
    GPT-2にレビューの感情を答えさせるためのプロンプトを作る。
    """

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
    """
    SST-2の数値ラベルを、GPT-2に生成させる文字列へ変換する。

    GPT-2では単語の前の空白もトークン化に影響するため、
    positive / negative の前に空白を付ける。
    """

    if label == 1:
        return " positive"

    return " negative"


# ============================================================
# DPO用の選好データセットを作る
# ============================================================

def create_preference_dataset(
    texts: list[str],
    labels: list[int],
    tokenizer,
) -> Dataset:
    """
    各SST-2データを、TRLのDPOTrainerが受け取る次の形式へ変換する。

    {
        "prompt":   共通の質問,
        "chosen":   望ましい応答,
        "rejected": 望ましくない応答,
    }

    例:
        正解ラベルがpositiveの場合

        prompt:
            ... Sentiment:

        chosen:
             positive

        rejected:
             negative
    """

    prompts: list[str] = []
    chosen_answers: list[str] = []
    rejected_answers: list[str] = []

    for sentence, label in zip(texts, labels):
        prompt = build_prompt(sentence)

        # 正解ラベルを含む応答
        chosen = label_to_answer(label)

        # 正解とは反対のラベルを含む応答
        rejected = label_to_answer(1 - label)

        # positive / negative を生成した後で終了することも学習させる
        chosen += tokenizer.eos_token
        rejected += tokenizer.eos_token

        prompts.append(prompt)
        chosen_answers.append(chosen)
        rejected_answers.append(rejected)

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosen_answers,
            "rejected": rejected_answers,
        }
    )


# ============================================================
# 生成結果からpositive / negativeを取り出す
# ============================================================

def extract_sentiment(generated_text: str) -> str:
    normalized = generated_text.strip().lower()

    if normalized.startswith("positive"):
        return "positive"

    if normalized.startswith("negative"):
        return "negative"

    return "unknown"


# ============================================================
# 開発データで生成精度を測る
# ============================================================

@torch.no_grad()
def evaluate_generation_accuracy(
    model,
    tokenizer,
    texts: list[str],
    labels: list[int],
    device: torch.device,
) -> float:
    """
    DPO後のモデルに実際にpositive / negativeを生成させ、
    SST-2開発データに対する正解率を求める。
    """

    model.eval()

    correct = 0

    for sentence, label in tqdm(
        zip(texts, labels),
        total=len(texts),
        desc="Generation accuracy",
    ):
        prompt = build_prompt(sentence)

        # プロンプトの末尾にある実際のレビューを残すため、
        # tokenizer.truncation_sideは"left"に設定してある
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        tokens = {
            key: value.to(device)
            for key, value in tokens.items()
        }

        generated_ids = model.generate(
            **tokens,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_length = tokens["input_ids"].size(1)

        answer_ids = generated_ids[
            0,
            prompt_length:,
        ]

        generated_text = tokenizer.decode(
            answer_ids,
            skip_special_tokens=True,
        )

        prediction = extract_sentiment(generated_text)

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
    # フォルダ
    # --------------------------------------------------------

    chapter10_dir = Path(__file__).resolve().parent

    sst2_dir = (
        chapter10_dir.parent
        / "chapter7"
        / "SST-2"
    )

    train_path = sst2_dir / "train.tsv"
    dev_path = sst2_dir / "dev.tsv"

    # 98番で保存したモデル
    sft_model_dir = (
        chapter10_dir
        / "98_gpt2_medium_sst2_batch8"
    )

    # DPOTrainerのチェックポイント保存先
    trainer_output_dir = (
        chapter10_dir
        / "99_dpo_checkpoints"
    )

    # 最終モデルの保存先
    final_model_dir = (
        chapter10_dir
        / "99_gpt2_medium_dpo"
    )

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            "SST-2が見つかりません。\n"
            f"train: {train_path}\n"
            f"dev:   {dev_path}"
        )

    # --------------------------------------------------------
    # DPOをどのモデルから始めるか
    # --------------------------------------------------------

    if START_FROM_98_MODEL:
        if not sft_model_dir.exists():
            raise FileNotFoundError(
                "START_FROM_98_MODEL=Trueですが、"
                "98番の学習済みモデルが見つかりません。\n"
                f"expected: {sft_model_dir}\n"
                "98番を先に実行するか、"
                "START_FROM_98_MODEL=Falseにしてください。"
            )

        model_source = str(sft_model_dir)

        print(
            "98番でファインチューニングしたモデルから"
            "DPOを開始します。"
        )
    else:
        model_source = MODEL_NAME

        print(
            "元のGPT-2-mediumから直接DPOを開始します。"
        )

    # --------------------------------------------------------
    # データの読み込み
    # --------------------------------------------------------

    train_texts, train_labels = load_sst2(
        train_path
    )

    dev_texts, dev_labels = load_sst2(
        dev_path
    )

    if DEBUG:
        train_texts = train_texts[
            :DEBUG_TRAIN_SIZE
        ]
        train_labels = train_labels[
            :DEBUG_TRAIN_SIZE
        ]

        dev_texts = dev_texts[
            :DEBUG_DEV_SIZE
        ]
        dev_labels = dev_labels[
            :DEBUG_DEV_SIZE
        ]

    print(f"train size: {len(train_texts)}")
    print(f"dev size:   {len(dev_texts)}")

    # --------------------------------------------------------
    # GPUと精度
    # --------------------------------------------------------

    use_cuda = torch.cuda.is_available()

    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    # 対応GPUではbf16を優先する。
    # bf16非対応のCUDA GPUではfp16を使う。
    use_bf16 = (
        use_cuda
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    use_fp16 = use_cuda and not use_bf16

    if use_bf16:
        model_dtype = torch.bfloat16
    elif use_fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    print(f"device: {device}")
    print(f"dtype:  {model_dtype}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_source
    )

    # GPT-2には専用のpad tokenがないので、
    # eos tokenをpaddingにも利用する
    tokenizer.pad_token = tokenizer.eos_token

    # DPOTrainerでは左paddingを使用する
    tokenizer.padding_side = "left"

    # 長すぎる場合は末尾の実レビューを残す
    tokenizer.truncation_side = "left"

    # --------------------------------------------------------
    # DPO用データセット
    # --------------------------------------------------------

    train_dataset = create_preference_dataset(
        train_texts,
        train_labels,
        tokenizer,
    )

    dev_dataset = create_preference_dataset(
        dev_texts,
        dev_labels,
        tokenizer,
    )

    print("\nDPOデータの例")
    print("prompt:")
    print(train_dataset[0]["prompt"])
    print("chosen:")
    print(repr(train_dataset[0]["chosen"]))
    print("rejected:")
    print(repr(train_dataset[0]["rejected"]))

    # --------------------------------------------------------
    # 学習するpolicy model
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=model_dtype,
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # 学習中は生成用キャッシュを使わない
    model.config.use_cache = False

    # --------------------------------------------------------
    # DPOの学習設定
    # --------------------------------------------------------

    training_args = DPOConfig(
        output_dir=str(trainer_output_dir),

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=LEARNING_RATE,

        # DPO固有の設定
        beta=BETA,
        loss_type="sigmoid",
        max_length=MAX_LENGTH,

        # 毎エポック、開発データで評価する
        eval_strategy="epoch",

        # 毎エポック、モデルを保存する
        save_strategy="epoch",

        # 開発lossが最も小さいモデルを最後に読み戻す
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=1,

        logging_steps=10,
        logging_first_step=True,

        # GPUに応じてbf16またはfp16を使用
        bf16=use_bf16,
        fp16=use_fp16,

        # 速度を優先し、gradient checkpointingは使わない
        gradient_checkpointing=False,

        # 幅広い環境で動かしやすいOptimizer
        optim="adamw_torch",

        report_to="none",
        seed=SEED,
    )

    # --------------------------------------------------------
    # DPOTrainer
    # --------------------------------------------------------

    trainer = DPOTrainer(
        # DPOで更新するpolicy model
        model=model,

        # Noneの場合、TRLが学習開始時点のmodelを
        # reference modelとして用意する
        ref_model=None,

        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=dev_dataset,

        # TokenizerをTRLへ渡す
        processing_class=tokenizer,
    )

    # --------------------------------------------------------
    # DPO学習
    # --------------------------------------------------------

    trainer.train()

    # 最良モデルが読み戻された状態で評価
    eval_metrics = trainer.evaluate()

    print("\nDPO evaluation metrics")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # 最終モデルを保存
    # --------------------------------------------------------

    trainer.save_model(
        str(final_model_dir)
    )

    tokenizer.save_pretrained(
        final_model_dir
    )

    print(
        f"\nDPO model saved: {final_model_dir}"
    )

    # --------------------------------------------------------
    # 生成評価のため、一度GPUメモリを整理する
    # --------------------------------------------------------

    del trainer
    del model

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # 保存したDPOモデルを読み直す
    # --------------------------------------------------------

    dpo_model = (
        AutoModelForCausalLM.from_pretrained(
            final_model_dir,
            dtype=model_dtype,
        )
    )

    dpo_model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # 生成時はキャッシュを利用する
    dpo_model.config.use_cache = True

    dpo_model.to(device)

    # --------------------------------------------------------
    # 実際にpositive / negativeを生成して正解率を測る
    # --------------------------------------------------------

    dev_accuracy = evaluate_generation_accuracy(
        dpo_model,
        tokenizer,
        dev_texts,
        dev_labels,
        device,
    )

    print(
        f"generation dev accuracy: "
        f"{dev_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()

#出力
# 98番でファインチューニングしたモデルからDPOを開始します。
# train size: 512
# dev size:   100
# device: cuda
# dtype:  torch.bfloat16

# DPOデータの例
# prompt:
# Classify the sentiment of the movie review as positive or negative.

# Review: I loved this movie.
# Sentiment: positive

# Review: This movie was boring and terrible.
# Sentiment: negative

# Review: hide new secretions from the parental units 
# Sentiment:
# chosen:
# ' negative<|endoftext|>'
# rejected:
# ' positive<|endoftext|>'
# Adding EOS to train dataset: 100%|█████████████████████████████████| 512/512 [00:00<00:00, 1925.27 examples/s]
# Tokenizing train dataset: 100%|████████████████████████████████████| 512/512 [00:00<00:00, 1383.32 examples/s]
# Adding EOS to eval dataset: 100%|█████████████████████████████████| 100/100 [00:00<00:00, 15387.42 examples/s]
# Tokenizing eval dataset: 100%|█████████████████████████████████████| 100/100 [00:00<00:00, 1353.92 examples/s]
# [RANK 0] Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
# {'loss': 0.6937, 'grad_norm': 1.515625, 'learning_rate': 1e-06, 'entropy': 0.03660983964800835, 'num_tokens': 546.0, 'logits/chosen': 31.236661911010742, 'logits/rejected': 27.201839447021484, 'mean_token_accuracy': 1.0, 'rewards/chosen': -3.852844383800402e-05, 'rewards/rejected': 0.0011333584552630782, 'rewards/accuracies': 0.0, 'rewards/margins': -0.001171886920928955, 'logps/chosen': -0.01557457447052002, 'logps/rejected': -4.8827619552612305, 'epoch': 0.01}
# {'loss': 0.6932, 'grad_norm': 1.4296875, 'learning_rate': 9.296874999999999e-07, 'entropy': 0.056635118607017726, 'num_tokens': 5200.0, 'logits/chosen': 33.34497324625651, 'logits/rejected': 29.672111723158096, 'mean_token_accuracy': 0.9722222222222222, 'rewards/chosen': -8.201532641452205e-05, 'rewards/rejected': -4.691878469505658e-05, 'rewards/accuracies': 0.3611111111111111, 'rewards/margins': -3.509656138097247e-05, 'logps/chosen': -0.2825794484880235, 'logps/rejected': -4.494886822170681, 'epoch': 0.08}
# {'loss': 0.6932, 'grad_norm': 0.76953125, 'learning_rate': 8.515624999999999e-07, 'entropy': 0.05816607140004635, 'num_tokens': 10194.0, 'logits/chosen': 34.949266052246095, 'logits/rejected': 30.83035316467285, 'mean_token_accuracy': 0.975, 'rewards/chosen': -9.293020205518588e-05, 'rewards/rejected': 0.00010421151819173247, 'rewards/accuracies': 0.35, 'rewards/margins': -0.00019714177760761232, 'logps/chosen': -0.11343007683753967, 'logps/rejected': -4.670412588119507, 'epoch': 0.16}
# {'loss': 0.6932, 'grad_norm': 2.0, 'learning_rate': 7.734375e-07, 'entropy': 0.034575009858235715, 'num_tokens': 15360.0, 'logits/chosen': 33.97690200805664, 'logits/rejected': 29.520367240905763, 'mean_token_accuracy': 1.0, 'rewards/chosen': 1.2956560186694333e-05, 'rewards/rejected': 7.551730668637901e-05, 'rewards/accuracies': 0.375, 'rewards/margins': -6.256073247641325e-05, 'logps/chosen': -0.016891199350357055, 'logps/rejected': -5.315325593948364, 'epoch': 0.23}
# {'loss': 0.6932, 'grad_norm': 1.3828125, 'learning_rate': 6.953125e-07, 'entropy': 0.051329069398343564, 'num_tokens': 20430.0, 'logits/chosen': 34.73467102050781, 'logits/rejected': 31.348983573913575, 'mean_token_accuracy': 0.9875, 'rewards/chosen': 3.479838122188994e-05, 'rewards/rejected': 8.34685459267348e-05, 'rewards/accuracies': 0.35, 'rewards/margins': -4.867018142249435e-05, 'logps/chosen': -0.05377461910247803, 'logps/rejected': -5.100838947296142, 'epoch': 0.31}
# {'loss': 0.6931, 'grad_norm': 1.5625, 'learning_rate': 6.171875e-07, 'entropy': 0.07539900317788124, 'num_tokens': 25570.0, 'logits/chosen': 33.816138458251956, 'logits/rejected': 30.046007347106933, 'mean_token_accuracy': 0.975, 'rewards/chosen': -1.0171532960612239e-05, 'rewards/rejected': -0.0001737457641866058, 'rewards/accuracies': 0.475, 'rewards/margins': 0.00016357420827262104, 'logps/chosen': -0.08472210764884949, 'logps/rejected': -4.695085835456848, 'epoch': 0.39}
# {'loss': 0.6934, 'grad_norm': 0.41796875, 'learning_rate': 5.390625e-07, 'entropy': 0.04041256196796894, 'num_tokens': 30682.0, 'logits/chosen': 34.026074981689455, 'logits/rejected': 30.929283332824706, 'mean_token_accuracy': 0.975, 'rewards/chosen': -6.687044742648141e-06, 'rewards/rejected': 0.0004815918258827878, 'rewards/accuracies': 0.225, 'rewards/margins': -0.000488278862030711, 'logps/chosen': -0.213075453042984, 'logps/rejected': -4.811097979545593, 'epoch': 0.47}
# {'loss': 0.6928, 'grad_norm': 2.28125, 'learning_rate': 4.609375e-07, 'entropy': 0.03545398637652397, 'num_tokens': 36028.0, 'logits/chosen': 33.42469673156738, 'logits/rejected': 29.892803573608397, 'mean_token_accuracy': 1.0, 'rewards/chosen': 4.1263104139943604e-05, 'rewards/rejected': -0.0007424282972351648, 'rewards/accuracies': 0.55, 'rewards/margins': 0.0007836914068320766, 'logps/chosen': -0.02673887610435486, 'logps/rejected': -5.7155327796936035, 'epoch': 0.55}
# {'loss': 0.6932, 'grad_norm': 1.2421875, 'learning_rate': 3.828125e-07, 'entropy': 0.03214753335341811, 'num_tokens': 41140.0, 'logits/chosen': 35.18330764770508, 'logits/rejected': 31.384659385681154, 'mean_token_accuracy': 1.0, 'rewards/chosen': 1.2385249328872306e-05, 'rewards/rejected': 4.951179726049304e-05, 'rewards/accuracies': 0.4, 'rewards/margins': -3.712654579430818e-05, 'logps/chosen': -0.013809871673583985, 'logps/rejected': -5.046382713317871, 'epoch': 0.62}
# {'loss': 0.6931, 'grad_norm': 1.2109375, 'learning_rate': 3.046875e-07, 'entropy': 0.047300076484680174, 'num_tokens': 46408.0, 'logits/chosen': 33.3373197555542, 'logits/rejected': 29.70691204071045, 'mean_token_accuracy': 0.9875, 'rewards/chosen': -4.736572586807597e-05, 'rewards/rejected': -0.00013891996350139378, 'rewards/accuracies': 0.425, 'rewards/margins': 9.155424195341765e-05, 'logps/chosen': -0.10548117160797119, 'logps/rejected': -4.707312273979187, 'epoch': 0.7}
# {'loss': 0.6932, 'grad_norm': 2.421875, 'learning_rate': 2.265625e-07, 'entropy': 0.058323978818953036, 'num_tokens': 51584.0, 'logits/chosen': 34.32907638549805, 'logits/rejected': 31.101899719238283, 'mean_token_accuracy': 0.975, 'rewards/chosen': -0.00018751562550960442, 'rewards/rejected': -0.00011732278362615034, 'rewards/accuracies': 0.45, 'rewards/margins': -7.019282056717203e-05, 'logps/chosen': -0.10340771079063416, 'logps/rejected': -4.676668190956116, 'epoch': 0.78}
# {'loss': 0.6933, 'grad_norm': 0.89453125, 'learning_rate': 1.4843749999999998e-07, 'entropy': 0.05008765775710344, 'num_tokens': 56792.0, 'logits/chosen': 33.94680480957031, 'logits/rejected': 30.537310981750487, 'mean_token_accuracy': 0.9875, 'rewards/chosen': -2.1896362298434724e-05, 'rewards/rejected': 0.00018348935438723403, 'rewards/accuracies': 0.325, 'rewards/margins': -0.00020538569224299864, 'logps/chosen': -0.13150500059127807, 'logps/rejected': -4.641343975067139, 'epoch': 0.86}
# {'loss': 0.693, 'grad_norm': 0.98828125, 'learning_rate': 7.03125e-08, 'entropy': 0.04310357607901096, 'num_tokens': 61918.0, 'logits/chosen': 34.72108955383301, 'logits/rejected': 30.67993927001953, 'mean_token_accuracy': 0.9875, 'rewards/chosen': -2.8057097517830697e-05, 'rewards/rejected': -0.00026426136755617333, 'rewards/accuracies': 0.475, 'rewards/margins': 0.00023620428255526348, 'logps/chosen': -0.05762381553649902, 'logps/rejected': -4.990210485458374, 'epoch': 0.94}
# {'eval_loss': 0.6933920383453369, 'eval_runtime': 4.9896, 'eval_samples_per_second': 20.042, 'eval_steps_per_second': 5.01, 'eval_entropy': 0.05862738043069839, 'eval_num_tokens': 66172.0, 'eval_logits/chosen': 30.60683692932129, 'eval_logits/rejected': 27.671305694580077, 'eval_mean_token_accuracy': 0.96, 'eval_rewards/chosen': -4.030729342957784e-05, 'eval_rewards/rejected': 0.0004484008307917975, 'eval_rewards/accuracies': 0.28, 'eval_rewards/margins': -0.0004887081382912584, 'eval_logps/chosen': -0.3105681636929512, 'eval_logps/rejected': -4.6560815334320065, 'epoch': 1.0}
# 100%|███████████████████████████████████████████████████████████████████████| 128/128 [00:56<00:00,  2.70it/s]There were missing keys in the checkpoint model loaded: ['lm_head.weight'].                                    
# {'train_runtime': 104.0036, 'train_samples_per_second': 4.923, 'train_steps_per_second': 1.231, 'train_loss': 0.6931477868929505, 'epoch': 1.0}
# 100%|███████████████████████████████████████████████████████████████████████| 128/128 [01:44<00:00,  1.23it/s]
# 100%|█████████████████████████████████████████████████████████████████████████| 25/25 [00:04<00:00,  5.34it/s]

# DPO evaluation metrics
# eval_loss: 0.6933920383453369
# eval_runtime: 4.8927
# eval_samples_per_second: 20.439
# eval_steps_per_second: 5.11
# eval_entropy: 0.05862738043069839
# eval_num_tokens: 66172.0
# eval_logits/chosen: 30.60683692932129
# eval_logits/rejected: 27.671305694580077
# eval_mean_token_accuracy: 0.96
# eval_rewards/chosen: -4.030729342957784e-05
# eval_rewards/rejected: 0.0004484008307917975
# eval_rewards/accuracies: 0.28
# eval_rewards/margins: -0.0004887081382912584
# eval_logps/chosen: -0.3105681636929512
# eval_logps/rejected: -4.6560815334320065
# epoch: 1.0

# Generation accuracy: 100%|██████████████████████████████████████████████████| 100/100 [00:06<00:00, 14.40it/s]
# generation dev accuracy: 0.9200