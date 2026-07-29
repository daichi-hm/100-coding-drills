from __future__ import annotations

from pathlib import Path
from types import ModuleType
import html
import importlib.util
import re

import torch
from torch import nn
from tqdm import tqdm
from sacrebleu.metrics import BLEU


# ============================================================
# 1. ファイルの場所
# ============================================================

# 93.pyが置かれているフォルダ
BASE_DIR = Path(__file__).resolve().parent

# 91番のモデルクラスを再利用するために読み込むファイル
PROBLEM91_PATH = BASE_DIR / "91.py"

# 90番で作った評価データ
#
# test.ja:
#   形態素単位に分割済みの日本語文
#
# test.en:
#   単語単位に分割済みの正解英訳
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TEST_JA_PATH = PROCESSED_DIR / "test.ja"
TEST_EN_PATH = PROCESSED_DIR / "test.en"

# 90番で作った語彙表
VOCAB_DIR = BASE_DIR / "data" / "vocab"
JA_VOCAB_PATH = VOCAB_DIR / "ja_vocab.json"
EN_VOCAB_PATH = VOCAB_DIR / "en_vocab.json"

# 91番で開発データのlossが最小だったモデル
CHECKPOINT_PATH = (
    BASE_DIR
    / "models"
    / "91_best_model.pt"
)

# 93番の結果を保存する場所
RESULT_DIR = BASE_DIR / "results"
PREDICTION_PATH = (
    RESULT_DIR
    / "93_predictions.en"
)
REFERENCE_PATH = (
    RESULT_DIR
    / "93_references.en"
)


# ============================================================
# 2. 評価時の設定
# ============================================================

# <eos>が生成されない場合に備え、
# 1文につき最大で何トークンまで英語を生成するかを決める。
MAX_GENERATION_TOKENS = 256

# None:
#   評価データを全件使い、正式なBLEUを計算する。
#
# 10などの整数:
#   先頭の指定件数だけで動作確認する。
#   この場合のBLEUは正式な評価値ではない。
MAX_EVALUATION_SENTENCES: int | None = None


# ============================================================
# 3. 91.pyをモジュールとして読み込む
# ============================================================

def load_problem91_module(
    problem91_path: Path,
) -> ModuleType:
    """
    91.pyをproblem91という名前で読み込み、
    91番のTranslationTransformerとload_vocabularyを
    93.pyから利用できるようにする。
    """

    if not problem91_path.exists():
        raise FileNotFoundError(
            f"91.pyが見つかりません: "
            f"{problem91_path}"
        )

    # 「この場所の91.pyをproblem91という名前で読む」
    # ための情報を作る。
    spec = importlib.util.spec_from_file_location(
        "problem91",
        problem91_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"91.pyを読み込めません: "
            f"{problem91_path}"
        )

    # problem91というモジュールオブジェクトを作る。
    problem91 = importlib.util.module_from_spec(
        spec
    )

    # 91.pyを実行し、クラスや関数の定義を
    # problem91の中へ読み込む。
    spec.loader.exec_module(problem91)

    return problem91


# ============================================================
# 4. IDから英語トークンを引く辞書を作る
# ============================================================

def create_id_to_token(
    token_to_id: dict[str, int],
) -> dict[int, str]:
    """
    token -> ID の辞書を、
    ID -> token の辞書へ反転する。

    例:
        {"Kyoto": 100, "is": 20}

        ↓

        {100: "Kyoto", 20: "is"}
    """

    return {
        token_id: token
        for token, token_id
        in token_to_id.items()
    }


# ============================================================
# 5. 91番の学習済みモデルを読み込む
# ============================================================

def load_trained_model(
    problem91: ModuleType,
    device: torch.device,
) -> nn.Module:
    """
    91_best_model.ptを読み込み、
    91番で学習したTransformerを復元する。
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "91番の学習済みモデルが"
            "見つかりません。\n"
            f"確認した場所: {CHECKPOINT_PATH}"
        )

    # GPU上で保存されたTensorも、
    # まずCPUへ読み込む。
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    # 91番では、モデルを再現するための設定を
    # checkpointへ保存している。
    model_config = checkpoint["model_config"]
    special_token_ids = checkpoint[
        "special_token_ids"
    ]

    # 91.pyに定義したTranslationTransformerを使い、
    # 学習時と同じ構造のモデルを作る。
    #
    # この時点では構造だけが同じで、
    # 重みはまだ学習済みの値ではない。
    model = problem91.TranslationTransformer(
        japanese_vocab_size=checkpoint[
            "japanese_vocab_size"
        ],
        english_vocab_size=checkpoint[
            "english_vocab_size"
        ],
        japanese_pad_id=special_token_ids[
            "japanese_pad_id"
        ],
        english_pad_id=special_token_ids[
            "english_pad_id"
        ],
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_encoder_layers=model_config[
            "num_encoder_layers"
        ],
        num_decoder_layers=model_config[
            "num_decoder_layers"
        ],
        dim_feedforward=model_config[
            "dim_feedforward"
        ],
        dropout=model_config["dropout"],
        max_position_length=model_config[
            "max_position_length"
        ],
    )

    # 91番で学習した重みを、
    # 上で作ったモデルへ読み込む。
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # GPUへ移動し、推論モードへ切り替える。
    model.to(device)
    model.eval()

    print(
        "読み込んだモデルのエポック:",
        checkpoint["epoch"],
    )

    print(
        "そのモデルのvalid loss:",
        f"{checkpoint['valid_loss']:.4f}",
    )

    return model


# ============================================================
# 6. 英語トークン列を読みやすい文へ戻す
# ============================================================

def detokenize_english(
    tokens: list[str],
) -> str:
    """
    空白区切りの英語トークンを、
    sacreBLEUへ渡せる通常の英文へ戻す。

    例:
        ["Kyoto", "is", "beautiful", "."]

        ↓

        "Kyoto is beautiful."
    """

    text = " ".join(tokens)

    # &amp;や&quot;などを通常の記号へ戻す。
    text = html.unescape(text)

    # state @-@ of @-@ the @-@ art
    #   ↓
    # state-of-the-art
    text = text.replace(" @-@ ", "-")

    # do n't -> don't
    text = text.replace(" n't", "n't")

    # it 's -> it's など
    text = re.sub(
        r"\s+'(s|re|ve|m|d|ll)\b",
        r"'\1",
        text,
    )

    # ピリオドやカンマなどの前の空白を削除する。
    text = re.sub(
        r"\s+([.,!?;:%\)\]\}])",
        r"\1",
        text,
    )

    # 開き括弧の直後の空白を削除する。
    text = re.sub(
        r"([\(\[\{])\s+",
        r"\1",
        text,
    )

    # 連続する空白を1個にする。
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 7. トークン化済み日本語1文を翻訳する
# ============================================================

@torch.no_grad()
def translate_tokenized_japanese(
    model: nn.Module,
    japanese_line: str,
    japanese_token_to_id: dict[str, int],
    english_token_to_id: dict[str, int],
    english_id_to_token: dict[int, str],
    device: torch.device,
    max_generation_tokens: int,
) -> str:
    """
    90番で形態素分割済みの日本語1文を、
    Greedy Decodingで英語へ翻訳する。

    92番との違い:
        92番は任意入力なのでMeCabで分割した。
        93番のtest.jaは90番で分割済みなので、
        split()するだけでよい。
    """

    # test.jaは、たとえば
    #
    #   京都 は 美しい 都市 です 。
    #
    # のように空白区切りで保存されている。
    japanese_tokens = japanese_line.split()

    if not japanese_tokens:
        raise ValueError(
            "評価データに空の日本語文があります。"
        )

    japanese_unk_id = japanese_token_to_id[
        "<unk>"
    ]
    japanese_bos_id = japanese_token_to_id[
        "<bos>"
    ]
    japanese_eos_id = japanese_token_to_id[
        "<eos>"
    ]

    # 日本語トークンをIDへ変換する。
    # 語彙表にないトークンは<unk>にする。
    japanese_ids = [
        japanese_token_to_id.get(
            token,
            japanese_unk_id,
        )
        for token in japanese_tokens
    ]

    # 文頭と文末の特殊トークンを追加する。
    japanese_ids = (
        [japanese_bos_id]
        + japanese_ids
        + [japanese_eos_id]
    )

    # 位置エンコーディングが扱える長さを超える場合は、
    # モデルへ入力できないので停止する。
    max_position_length = (
        model.positional_encoding
        .positional_encoding
        .size(1)
    )

    if len(japanese_ids) > max_position_length:
        raise ValueError(
            f"日本語文が長すぎます: "
            f"{len(japanese_ids)}トークン\n"
            f"モデルの上限: "
            f"{max_position_length}トークン"
        )

    # shape:
    #   [日本語文長]
    #       ↓
    #   [1, 日本語文長]
    #
    # 先頭の1はバッチサイズ。
    source_ids = torch.tensor(
        [japanese_ids],
        dtype=torch.long,
        device=device,
    )

    english_pad_id = english_token_to_id[
        "<pad>"
    ]
    english_bos_id = english_token_to_id[
        "<bos>"
    ]
    english_eos_id = english_token_to_id[
        "<eos>"
    ]

    # 英語生成は<bos>から始める。
    generated_ids = [english_bos_id]

    # 最大生成長まで、1トークンずつ生成する。
    for _ in range(max_generation_tokens):
        decoder_input_ids = torch.tensor(
            [generated_ids],
            dtype=torch.long,
            device=device,
        )

        # logitsのshape:
        # [1, 現在の英語文長, 英語語彙数]
        logits = model(
            japanese_ids=source_ids,
            decoder_input_ids=decoder_input_ids,
        )

        # 最後の位置における、
        # 次の英語トークン候補のスコアを取り出す。
        next_token_logits = (
            logits[0, -1].clone()
        )

        # 文生成の途中で<pad>や<bos>は不要なので、
        # 選ばれないようにスコアを負の無限大にする。
        next_token_logits[
            english_pad_id
        ] = -float("inf")

        next_token_logits[
            english_bos_id
        ] = -float("inf")

        # 最もスコアが高いトークンIDを選ぶ。
        next_token_id = int(
            torch.argmax(
                next_token_logits
            ).item()
        )

        generated_ids.append(
            next_token_id
        )

        # <eos>なら文の生成を終了する。
        if next_token_id == english_eos_id:
            break

    # <bos>、<eos>、<pad>は最終的な英文に不要。
    special_ids = {
        english_pad_id,
        english_bos_id,
        english_eos_id,
    }

    generated_tokens = [
        english_id_to_token.get(
            token_id,
            "<unk>",
        )
        for token_id in generated_ids
        if token_id not in special_ids
    ]

    # トークンのリストを通常の英文へ戻す。
    return detokenize_english(
        generated_tokens
    )


# ============================================================
# 8. 評価データを読み込む
# ============================================================

def load_evaluation_data(
    japanese_path: Path,
    english_path: Path,
    max_sentences: int | None,
) -> tuple[list[str], list[str]]:
    """
    test.jaとtest.enを読み込む。

    戻り値:
        japanese_lines:
            トークン化済み日本語文のリスト

        reference_sentences:
            detokenize済み正解英訳のリスト
    """

    if not japanese_path.exists():
        raise FileNotFoundError(
            f"評価用日本語データが"
            f"見つかりません: {japanese_path}"
        )

    if not english_path.exists():
        raise FileNotFoundError(
            f"評価用英語データが"
            f"見つかりません: {english_path}"
        )

    with japanese_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        japanese_lines = file.read().splitlines()

    with english_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        tokenized_english_lines = (
            file.read().splitlines()
        )

    if len(japanese_lines) != len(
        tokenized_english_lines
    ):
        raise ValueError(
            "test.jaとtest.enの行数が"
            "一致しません。\n"
            f"日本語: {len(japanese_lines):,}\n"
            f"英語: {len(tokenized_english_lines):,}"
        )

    # 動作確認用の件数制限。
    if max_sentences is not None:
        japanese_lines = japanese_lines[
            :max_sentences
        ]
        tokenized_english_lines = (
            tokenized_english_lines[
                :max_sentences
            ]
        )

    # test.enも90番で単語単位に分割済みなので、
    # sacreBLEUへ渡す前に通常の英文へ戻す。
    reference_sentences = [
        detokenize_english(
            line.split()
        )
        for line in tokenized_english_lines
    ]

    return japanese_lines, reference_sentences


# ============================================================
# 9. 文字列のリストをファイルへ保存する
# ============================================================

def save_lines(
    output_path: Path,
    lines: list[str],
) -> None:
    """
    1文を1行としてUTF-8のテキストへ保存する。
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for line in lines:
            file.write(line + "\n")


# ============================================================
# 10. メイン処理
# ============================================================

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA対応GPUを利用できません。\n"
            "GPUノードに接続しているか"
            "確認してください。"
        )

    device = torch.device("cuda")

    print(f"使用デバイス: {device}")
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------------
    # 91.pyと語彙表を読み込む
    # --------------------------------------------------------

    problem91 = load_problem91_module(
        PROBLEM91_PATH
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
        create_id_to_token(
            english_token_to_id
        )
    )

    print(
        "日本語語彙数:",
        f"{len(japanese_token_to_id):,}",
    )
    print(
        "英語語彙数:",
        f"{len(english_token_to_id):,}",
    )

    # --------------------------------------------------------
    # 学習済みモデルを読み込む
    # --------------------------------------------------------

    model = load_trained_model(
        problem91=problem91,
        device=device,
    )

    # --------------------------------------------------------
    # 評価データを読み込む
    # --------------------------------------------------------

    (
        japanese_lines,
        reference_sentences,
    ) = load_evaluation_data(
        japanese_path=TEST_JA_PATH,
        english_path=TEST_EN_PATH,
        max_sentences=(
            MAX_EVALUATION_SENTENCES
        ),
    )

    print(
        "評価する文数:",
        f"{len(japanese_lines):,}",
    )

    if MAX_EVALUATION_SENTENCES is not None:
        print(
            "注意: 一部データだけでの動作確認です。"
            "正式なBLEUではありません。"
        )

    # --------------------------------------------------------
    # 評価データを1文ずつ翻訳する
    # --------------------------------------------------------

    hypotheses: list[str] = []

    for japanese_line in tqdm(
        japanese_lines,
        desc="Translating test data",
    ):
        translated_sentence = (
            translate_tokenized_japanese(
                model=model,
                japanese_line=japanese_line,
                japanese_token_to_id=(
                    japanese_token_to_id
                ),
                english_token_to_id=(
                    english_token_to_id
                ),
                english_id_to_token=(
                    english_id_to_token
                ),
                device=device,
                max_generation_tokens=(
                    MAX_GENERATION_TOKENS
                ),
            )
        )

        hypotheses.append(
            translated_sentence
        )

    # --------------------------------------------------------
    # 翻訳結果と正解訳を保存する
    # --------------------------------------------------------

    save_lines(
        PREDICTION_PATH,
        hypotheses,
    )

    save_lines(
        REFERENCE_PATH,
        reference_sentences,
    )

    # --------------------------------------------------------
    # sacreBLEUでコーパス全体のBLEUを計算する
    # --------------------------------------------------------

    # tokenize="13a"はsacreBLEUの標準的な
    # 英語用Tokenizerである。
    bleu = BLEU(tokenize="13a")

    # hypotheses:
    #   モデルが生成した英文のリスト
    #
    # [reference_sentences]:
    #   正解訳の集合をさらにリストで囲んだもの。
    #   今回は各日本語文につき正解訳が1個なので、
    #   外側のリストの要素数は1になる。
    bleu_result = bleu.corpus_score(
        hypotheses,
        [reference_sentences],
    )

    print("\n============================")
    print("93番: BLEU評価結果")
    print("============================")
    print(bleu_result)
    print("BLEU score:", f"{bleu_result.score:.2f}")
    print("signature:", bleu.get_signature())

    print("\n翻訳結果:", PREDICTION_PATH)
    print("正解訳:", REFERENCE_PATH)

    # 最初の3例を表示する。
    number_of_examples = min(
        3,
        len(hypotheses),
    )

    print("\n翻訳例")

    for index in range(number_of_examples):
        print(f"\n例{index + 1}")
        print("日本語:", japanese_lines[index])
        print("モデル:", hypotheses[index])
        print("正解:  ", reference_sentences[index])


if __name__ == "__main__":
    main()

#出力
# 使用デバイス: cuda
# GPU: NVIDIA RTX A6000
# 日本語語彙数: 119,002
# 英語語彙数: 215,627
# 読み込んだモデルのエポック: 8
# そのモデルのvalid loss: 3.0717
# 評価する文数: 1,160
# Translating test data: 100%|██████████████████████████████████████████████| 1160/1160 [03:00<00:00,  6.44it/s]

# ============================
# 93番: BLEU評価結果
# ============================
# BLEU = 19.33 49.9/24.1/13.7/8.4 (BP = 1.000 ratio = 1.042 hyp_len = 27876 ref_len = 26765)
# BLEU score: 19.33
# signature: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0

# 翻訳例

# 例1
# 日本語: InfoboxBuddhist
# モデル: Spring:
# 正解:   Infobox Buddhist

# 例2
# 日本語: 道元 （ どう げん ） は 、 鎌倉 時代 初期 の 禅僧 。
# モデル: Dogen was a Zen priest in the early Kamakura period.
# 正解:   Dogen was a Zen monk in the early Kamakura period.

# 例3
# 日本語: 曹洞 宗 の 開祖 。
# モデル: He was the founder of the Soto sect.
# 正解:   The founder of Soto Zen