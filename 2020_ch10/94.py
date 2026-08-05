from __future__ import annotations

import csv
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import torch
from sacrebleu.metrics import BLEU
from torch import nn
from tqdm import tqdm


# ============================================================
# 設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROBLEM91_PATH = BASE_DIR / "91.py"
PROBLEM93_PATH = BASE_DIR / "93.py"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEV_JA_PATH = PROCESSED_DIR / "dev.ja"
DEV_EN_PATH = PROCESSED_DIR / "dev.en"

VOCAB_DIR = BASE_DIR / "data" / "vocab"
JA_VOCAB_PATH = VOCAB_DIR / "ja_vocab.json"
EN_VOCAB_PATH = VOCAB_DIR / "en_vocab.json"

RESULT_DIR = BASE_DIR / "results"
CSV_PATH = RESULT_DIR / "94_beam_bleu.csv"
PLOT_PATH = RESULT_DIR / "94_beam_bleu.png"

# 1から100程度までビーム幅を変える。
BEAM_WIDTHS = [1, 2, 4, 8, 16, 32, 64, 100]

# <eos>が生成されなかった場合の生成上限。
MAX_GENERATION_TOKENS = 256

# Noneなら開発セット全件を使う。
# 動作確認時だけ10などに変更する。
MAX_EVALUATION_SENTENCES: int | None = None


# 数字から始まる93.pyは通常のimportができないため、
# 動的に読み込む。
def load_module(
    name: str,
    path: Path,
) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(
            f"ファイルが見つかりません: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"ファイルを読み込めません: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@dataclass
class Beam:
    """
    ビーム探索で保持する1つの翻訳候補。
    """

    # <bos>を含む、ここまでの英語トークンID列
    token_ids: list[int]

    # 各トークンの対数生成確率の合計
    sum_log_probability: float

    # 最後に<eos>を生成したか
    finished: bool

    @property
    def score(self) -> float:
        """
        対数生成確率の平均を計算する。
        """

        # <bos>はモデルが生成したトークンではないため、
        # 生成トークン数には含めない。
        generated_length = max(
            1,
            len(self.token_ids) - 1,
        )

        return (
            self.sum_log_probability
            / generated_length
        )


@torch.no_grad()
def encode_source(
    model: nn.Module,
    source_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    日本語文をEncoderへ1回だけ入力する。

    戻り値:
        memory:
            Encoderの出力

        source_padding_mask:
            日本語側の<pad>位置
    """

    source_padding_mask = source_ids.eq(
        model.japanese_pad_id
    )

    # 日本語IDをEmbeddingへ変換する。
    source_embeddings = (
        model.japanese_embedding(source_ids)
        * math.sqrt(model.d_model)
    )

    # トークンの位置情報を加える。
    source_embeddings = (
        model.positional_encoding(
            source_embeddings
        )
    )

    # GPUでは自動混合精度を使って、
    # 推論時の計算量とメモリ使用量を抑える。
    with torch.autocast(
        device_type=source_ids.device.type,
        dtype=torch.float16,
        enabled=source_ids.is_cuda,
    ):
        memory = model.transformer.encoder(
            source_embeddings,
            src_key_padding_mask=(
                source_padding_mask
            ),
        )

    return memory, source_padding_mask


@torch.no_grad()
def predict_next_log_probabilities(
    model: nn.Module,
    beams: list[Beam],
    memory: torch.Tensor,
    source_padding_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    未完成の複数候補をまとめてDecoderへ入力し、
    次の英語トークンの対数確率を返す。

    戻り値のshape:
        [未完成候補数, 英語語彙数]
    """

    # 各候補の英語ID列を1つのバッチにする。
    decoder_input_ids = torch.tensor(
        [
            beam.token_ids
            for beam in beams
        ],
        dtype=torch.long,
        device=device,
    )

    target_embeddings = (
        model.english_embedding(
            decoder_input_ids
        )
        * math.sqrt(model.d_model)
    )

    target_embeddings = (
        model.positional_encoding(
            target_embeddings
        )
    )

    # Decoderが未来のトークンを見ないためのマスク。
    causal_mask = model.create_causal_mask(
        target_length=(
            decoder_input_ids.size(1)
        ),
        device=device,
    )

    batch_size = len(beams)

    # Encoder出力は全候補で共通なので、
    # 候補数だけ見かけ上コピーする。
    expanded_memory = memory.expand(
        batch_size,
        -1,
        -1,
    )

    expanded_source_mask = (
        source_padding_mask.expand(
            batch_size,
            -1,
        )
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(device.type == "cuda"),
    ):
        decoder_output = (
            model.transformer.decoder(
                tgt=target_embeddings,
                memory=expanded_memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=(
                    expanded_source_mask
                ),
            )
        )

        # 次の1単語を決めるため、
        # 最後の位置だけを英語語彙のスコアへ変換する。
        next_token_logits = (
            model.output_layer(
                decoder_output[:, -1]
            )
        )

    # softmaxではなくlog_softmaxを使うことで、
    # 各トークンの対数生成確率を得る。
    #
    # 数値を安定させるためfloat32で計算する。
    return torch.log_softmax(
        next_token_logits.float(),
        dim=-1,
    )


@torch.no_grad()
def translate_with_beam_search(
    model: nn.Module,
    japanese_line: str,
    japanese_token_to_id: dict[str, int],
    english_token_to_id: dict[str, int],
    english_id_to_token: dict[int, str],
    beam_width: int,
    device: torch.device,
    detokenize_english,
) -> str:
    """
    形態素分割済みの日本語1文を、
    ビーム探索で英語へ翻訳する。
    """

    if beam_width < 1:
        raise ValueError(
            "beam_widthは1以上にしてください。"
        )

    # --------------------------------------------------------
    # 日本語をIDへ変換する
    # --------------------------------------------------------

    japanese_unk_id = (
        japanese_token_to_id["<unk>"]
    )

    japanese_ids = [
        japanese_token_to_id.get(
            token,
            japanese_unk_id,
        )
        for token in japanese_line.split()
    ]

    japanese_ids = (
        [japanese_token_to_id["<bos>"]]
        + japanese_ids
        + [japanese_token_to_id["<eos>"]]
    )

    source_ids = torch.tensor(
        [japanese_ids],
        dtype=torch.long,
        device=device,
    )

    # 同じ日本語文のEncoder計算は1回だけ行う。
    memory, source_padding_mask = (
        encode_source(
            model,
            source_ids,
        )
    )

    pad_id = english_token_to_id["<pad>"]
    bos_id = english_token_to_id["<bos>"]
    eos_id = english_token_to_id["<eos>"]

    # 最初は<bos>だけを持つ候補が1つある。
    beams = [
        Beam(
            token_ids=[bos_id],
            sum_log_probability=0.0,
            finished=False,
        )
    ]

    # --------------------------------------------------------
    # 1トークンずつ生成する
    # --------------------------------------------------------

    for _ in range(
        MAX_GENERATION_TOKENS
    ):
        # <eos>をまだ生成していない候補だけを取り出す。
        active_beams = [
            beam
            for beam in beams
            if not beam.finished
        ]

        if not active_beams:
            break

        log_probabilities = (
            predict_next_log_probabilities(
                model=model,
                beams=active_beams,
                memory=memory,
                source_padding_mask=(
                    source_padding_mask
                ),
                device=device,
            )
        )

        # 文の途中に<pad>や<bos>は生成させない。
        log_probabilities[
            :,
            pad_id,
        ] = -float("inf")

        log_probabilities[
            :,
            bos_id,
        ] = -float("inf")

        # 各候補がここまでに持っている
        # 対数確率の合計をTensorにする。
        previous_sums = torch.tensor(
            [
                beam.sum_log_probability
                for beam in active_beams
            ],
            device=device,
        ).unsqueeze(1)

        # shape:
        # [現在の候補数, 英語語彙数]
        #
        # 各候補について、次の全単語を追加した場合の
        # 対数確率の合計を計算する。
        candidate_sums = (
            previous_sums
            + log_probabilities
        )

        vocabulary_size = (
            candidate_sums.size(1)
        )

        # 全ての
        #
        # 「現在の候補 × 次の英語単語」
        #
        # の中から、上位beam_width個を選ぶ。
        top_values, top_indices = (
            torch.topk(
                candidate_sums.flatten(),
                k=min(
                    beam_width,
                    candidate_sums.numel(),
                ),
            )
        )

        # flattenする前の、
        # どの親候補から生成されたかを求める。
        parent_indices = (
            top_indices
            // vocabulary_size
        )

        # 次に追加されたトークンIDを求める。
        next_token_ids = (
            top_indices
            % vocabulary_size
        )

        # 既に<eos>へ到達した候補は、
        # それ以上展開せず、そのまま残す。
        candidates = [
            beam
            for beam in beams
            if beam.finished
        ]

        # 上位の新しい候補を追加する。
        for (
            parent_index,
            token_id,
            sum_log_probability,
        ) in zip(
            parent_indices.tolist(),
            next_token_ids.tolist(),
            top_values.tolist(),
        ):
            parent = active_beams[
                parent_index
            ]

            candidates.append(
                Beam(
                    token_ids=(
                        parent.token_ids
                        + [token_id]
                    ),
                    sum_log_probability=(
                        sum_log_probability
                    ),
                    finished=(
                        token_id == eos_id
                    ),
                )
            )

        # 平均対数生成確率で並べ、
        # 上位beam_width個だけを次へ残す。
        beams = sorted(
            candidates,
            key=lambda beam: beam.score,
            reverse=True,
        )[:beam_width]

        if all(
            beam.finished
            for beam in beams
        ):
            break

    # 最後にもっともスコアが高い候補を採用する。
    best_beam = max(
        beams,
        key=lambda beam: beam.score,
    )

    special_ids = {
        pad_id,
        bos_id,
        eos_id,
    }

    generated_tokens = [
        english_id_to_token.get(
            token_id,
            "<unk>",
        )
        for token_id
        in best_beam.token_ids
        if token_id not in special_ids
    ]

    # 93番と同じ英語の後処理を使用する。
    return detokenize_english(
        generated_tokens
    )


def save_results(
    results: list[tuple[int, float]],
) -> None:
    """
    BLEUの途中結果をCSVとグラフへ保存する。
    """

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSVを保存する
    # --------------------------------------------------------

    with CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "beam_width",
            "bleu",
        ])

        writer.writerows(results)

    # --------------------------------------------------------
    # グラフを保存する
    # --------------------------------------------------------

    beam_widths = [
        beam_width
        for beam_width, _
        in results
    ]

    bleu_scores = [
        score
        for _, score
        in results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        beam_widths,
        bleu_scores,
        marker="o",
    )

    plt.xlabel("Beam width")
    plt.ylabel("BLEU")

    plt.title(
        "Beam width and BLEU "
        "on KFTT development set"
    )

    plt.xticks(beam_widths)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=150,
    )

    plt.close()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA対応GPU上で実行してください。"
        )

    device = torch.device("cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------------
    # 93番の処理を再利用する
    # --------------------------------------------------------

    problem93 = load_module(
        "problem93",
        PROBLEM93_PATH,
    )

    problem91 = (
        problem93.load_problem91_module(
            PROBLEM91_PATH
        )
    )

    japanese_token_to_id = (
        problem91.load_vocabulary(
            JA_VOCAB_PATH
        )
    )

    english_token_to_id = (
        problem91.load_vocabulary(
            EN_VOCAB_PATH
        )
    )

    english_id_to_token = (
        problem93.create_id_to_token(
            english_token_to_id
        )
    )

    model = problem93.load_trained_model(
        problem91=problem91,
        device=device,
    )

    # --------------------------------------------------------
    # 開発データを読み込む
    # --------------------------------------------------------

    (
        japanese_lines,
        reference_sentences,
    ) = problem93.load_evaluation_data(
        japanese_path=DEV_JA_PATH,
        english_path=DEV_EN_PATH,
        max_sentences=(
            MAX_EVALUATION_SENTENCES
        ),
    )

    print(
        "開発データ:",
        f"{len(japanese_lines):,}文",
    )

    bleu = BLEU(tokenize="13a")

    results: list[
        tuple[int, float]
    ] = []

    # --------------------------------------------------------
    # ビーム幅を変えながらBLEUを計算する
    # --------------------------------------------------------

    for beam_width in BEAM_WIDTHS:
        hypotheses: list[str] = []

        for japanese_line in tqdm(
            japanese_lines,
            desc=f"beam={beam_width}",
        ):
            translated_sentence = (
                translate_with_beam_search(
                    model=model,
                    japanese_line=(
                        japanese_line
                    ),
                    japanese_token_to_id=(
                        japanese_token_to_id
                    ),
                    english_token_to_id=(
                        english_token_to_id
                    ),
                    english_id_to_token=(
                        english_id_to_token
                    ),
                    beam_width=beam_width,
                    device=device,
                    detokenize_english=(
                        problem93
                        .detokenize_english
                    ),
                )
            )

            hypotheses.append(
                translated_sentence
            )

        bleu_result = bleu.corpus_score(
            hypotheses,
            [reference_sentences],
        )

        score = bleu_result.score

        results.append(
            (
                beam_width,
                score,
            )
        )

        # ビーム幅ごとに保存する。
        # 長時間実行が途中で止まっても、
        # 完了済みの結果はCSVに残る。
        save_results(results)

        print(
            f"beam width={beam_width:3d}, "
            f"BLEU={score:.2f}"
        )

    print("\nCSV:", CSV_PATH)
    print("グラフ:", PLOT_PATH)


if __name__ == "__main__":
    main()

#出力
# GPU: NVIDIA RTX A6000
# 読み込んだモデルのエポック: 8
# そのモデルのvalid loss: 3.0717
# 開発データ: 1,166文
# beam=1: 100%|█████████████████████████████████████████████████████████████| 1166/1166 [02:09<00:00,  9.02it/s]
# beam width=  1, BLEU=17.49
# beam=2: 100%|█████████████████████████████████████████████████████████████| 1166/1166 [02:22<00:00,  8.16it/s]
# beam width=  2, BLEU=18.15
# beam=4: 100%|█████████████████████████████████████████████████████████████| 1166/1166 [02:32<00:00,  7.66it/s]
# beam width=  4, BLEU=18.45
# beam=8: 100%|█████████████████████████████████████████████████████████████| 1166/1166 [02:42<00:00,  7.17it/s]
# beam width=  8, BLEU=18.44
# beam=16: 100%|████████████████████████████████████████████████████████████| 1166/1166 [02:58<00:00,  6.53it/s]
# beam width= 16, BLEU=18.27
# beam=32: 100%|████████████████████████████████████████████████████████████| 1166/1166 [03:24<00:00,  5.69it/s]
# beam width= 32, BLEU=18.11
# beam=64: 100%|████████████████████████████████████████████████████████████| 1166/1166 [04:19<00:00,  4.49it/s]
# beam width= 64, BLEU=17.85
# beam=100: 100%|███████████████████████████████████████████████████████████| 1166/1166 [05:30<00:00,  3.53it/s]
# beam width=100, BLEU=17.74