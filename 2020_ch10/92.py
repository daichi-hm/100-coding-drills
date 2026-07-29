from __future__ import annotations

from pathlib import Path
from types import ModuleType
import html
import importlib.util
import re

import MeCab
import torch
from torch import nn


# ============================================================
# ファイルの場所
# ============================================================

# 92.py自身が置かれているフォルダ
CHAPTER_DIR = Path(__file__).resolve().parent

# 91番のコード
#
# 92番ではモデル構造をもう一度書かず、
# 91.pyに定義したTranslationTransformerをそのまま利用する。
PROBLEM91_PATH = CHAPTER_DIR / "91.py"

# 90番で作成した語彙表
JA_VOCAB_PATH = (
    CHAPTER_DIR
    / "data"
    / "vocab"
    / "ja_vocab.json"
)

EN_VOCAB_PATH = (
    CHAPTER_DIR
    / "data"
    / "vocab"
    / "en_vocab.json"
)

# 91番でdev lossが最も小さかったモデル
CHECKPOINT_PATH = (
    CHAPTER_DIR
    / "models"
    / "91_best_model.pt"
)

# 91番では、特殊トークン込みで256トークンを超える
# 文対を学習データから除外した。
#
# 92番でも、モデルが学習時にほとんど見ていない
# 極端な長文は入力しないようにする。
MAX_SOURCE_LENGTH = 256

# <eos>が出なかった場合に備え、
# 最大で何トークンまで英語を生成するかを決める。
MAX_GENERATION_TOKENS = 100


# ============================================================
# 91.pyをPythonモジュールとして読み込む
# ============================================================

def load_problem91_module(
    problem91_path: Path,
) -> ModuleType:
    """
    91.pyをproblem91という名前で読み込み、
    クラスや関数を92.pyから利用できるようにする。
    """

    if not problem91_path.exists():
        raise FileNotFoundError(
            f"91.pyが見つかりません: {problem91_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "problem91",
        problem91_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"91.pyを読み込めません: "
            f"{problem91_path}"
        )

    problem91 = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(problem91)

    return problem91


# ============================================================
# 日本語入力を90番と同じように形態素分割する
# ============================================================

def create_mecab_tagger() -> MeCab.Tagger:
    """
    日本語を分かち書きするMeCabのTaggerを作る。

    90番の前処理でunidic-liteを使用したため、
    92番の入力にも同じ辞書を使用する。
    """

    try:
        import unidic_lite

        tagger = MeCab.Tagger(
            f"-d {unidic_lite.DICDIR} -Owakati"
        )

    except ImportError:
        # unidic-liteがない場合は、
        # その環境のMeCab既定辞書を使う。
        #
        # ただし、学習時と辞書が異なると分割結果も変わり、
        # <unk>が増える可能性がある。
        print(
            "注意: unidic_liteを読み込めなかったため、"
            "MeCabの既定辞書を使用します。"
        )

        tagger = MeCab.Tagger("-Owakati")

    # 環境によっては、最初に空文字列をparseしておくと
    # その後の解析が安定する。
    tagger.parse("")

    return tagger


def tokenize_japanese(
    text: str,
    tagger: MeCab.Tagger,
) -> list[str]:
    """
    入力された通常の日本語文を、
    形態素のリストへ変換する。

    例:
        "京都は美しい。"

        ↓

        ["京都", "は", "美しい", "。"]
    """

    parsed_text = tagger.parse(text)

    if parsed_text is None:
        raise RuntimeError(
            "MeCabによる日本語文の解析に失敗しました。"
        )

    return parsed_text.strip().split()


# ============================================================
# 語彙表の逆引き辞書を作る
# ============================================================

def create_id_to_token(
    token_to_id: dict[str, int],
) -> dict[int, str]:
    """
    91番のload_vocabulary()が返す辞書は、

        トークン -> ID

    の向きである。

    生成された英語IDを英単語へ戻すため、

        ID -> トークン

    の辞書も作る。
    """

    return {
        token_id: token
        for token, token_id in token_to_id.items()
    }


# ============================================================
# 91番の学習済みモデルを読み込む
# ============================================================

def load_trained_model(
    problem91: ModuleType,
    device: torch.device,
) -> nn.Module:
    """
    91番で保存したチェックポイントを読み込み、
    学習済みTransformerモデルを復元する。
    """

    # --------------------------------------------------------
    # 1. チェックポイントが存在するか確認する
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "91番の学習済みモデルが見つかりません。\n"
            f"確認した場所: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------------
    # 2. 91番で保存したチェックポイントを読み込む
    # --------------------------------------------------------

    # GPU上で保存したTensorも、まずCPUへ読み込む。
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    # --------------------------------------------------------
    # 3. 保存されていたモデル設定を取り出す
    # --------------------------------------------------------

    model_config = checkpoint["model_config"]

    special_token_ids = checkpoint[
        "special_token_ids"
    ]

    # --------------------------------------------------------
    # 4. 91番と同じ構造のモデルを作る
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5. 学習済みの重みをモデルへ入れる
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # 6. 使用するデバイスへ移し、推論モードにする
    # --------------------------------------------------------

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # 7. 読み込んだモデルの情報を表示する
    # --------------------------------------------------------

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
# 英語トークン列を読みやすい文へ戻す
# ============================================================

def detokenize_english(
    tokens: list[str],
) -> str:
    """
    空白区切りの英語トークンを簡易的に英文へ戻す。

    例:
        ["Kyoto", "is", "beautiful", "."]

        ↓

        "Kyoto is beautiful."
    """

    text = " ".join(tokens)

    # &apos;や&quot;などがあれば通常の記号へ戻す。
    text = html.unescape(text)

    # Moses系Tokenizerで現れることがある
    # "state @-@ of @-@ the @-@ art"を、
    # "state-of-the-art"へ近い形に戻す。
    text = text.replace(" @-@ ", "-")

    # do n't -> don't
    text = text.replace(" n't", "n't")

    # it 's -> it's など
    text = re.sub(
        r"\s+'(s|re|ve|m|d|ll)\b",
        r"'\1",
        text,
    )

    # 句読点の前の空白を削除する。
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
# Greedy Decodingで日本語1文を翻訳する
# ============================================================

@torch.no_grad()
def translate_greedily(
    model: nn.Module,
    japanese_text: str,
    japanese_tagger: MeCab.Tagger,
    japanese_token_to_id: dict[str, int],
    english_token_to_id: dict[str, int],
    english_id_to_token: dict[int, str],
    device: torch.device,
    max_generation_tokens: int,
) -> tuple[str, list[str], list[str]]:
    """
    91番の学習済みモデルを使い、
    日本語1文を英語へ翻訳する。

    Greedy Decodingでは、各生成ステップで
    最もスコアが高い英語トークンを1個選ぶ。
    """

    # --------------------------------------------------------
    # 1. 日本語文を形態素単位へ分割する
    # --------------------------------------------------------

    japanese_tokens = tokenize_japanese(
        japanese_text,
        japanese_tagger,
    )

    if not japanese_tokens:
        raise ValueError(
            "日本語文が空です。"
        )

    # --------------------------------------------------------
    # 2. 日本語トークンをIDへ変換する
    # --------------------------------------------------------

    japanese_unk_id = (
        japanese_token_to_id["<unk>"]
    )
    japanese_bos_id = (
        japanese_token_to_id["<bos>"]
    )
    japanese_eos_id = (
        japanese_token_to_id["<eos>"]
    )

    # 語彙表に存在するトークンなら、そのIDを使う。
    # 存在しない場合は<unk>のIDを使う。
    japanese_ids = [
        japanese_token_to_id.get(
            token,
            japanese_unk_id,
        )
        for token in japanese_tokens
    ]

    # 文頭に<bos>、文末に<eos>を追加する。
    japanese_ids = (
        [japanese_bos_id]
        + japanese_ids
        + [japanese_eos_id]
    )

    # 91番では256トークンを超える文対を除外して学習した。
    # ここでは文を途中で切らず、長すぎる場合はエラーにする。
    if len(japanese_ids) > MAX_SOURCE_LENGTH:
        raise ValueError(
            f"入力文は特殊トークン込みで"
            f"{len(japanese_ids)}トークンです。"
            f"{MAX_SOURCE_LENGTH}以下の文を"
            "入力してください。"
        )

    # [日本語文長]
    #     ↓
    # [1, 日本語文長]
    #
    # 先頭の1はバッチサイズである。
    source_ids = torch.tensor(
        [japanese_ids],
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # 3. 英語生成を<bos>から始める
    # --------------------------------------------------------

    english_pad_id = (
        english_token_to_id["<pad>"]
    )
    english_bos_id = (
        english_token_to_id["<bos>"]
    )
    english_eos_id = (
        english_token_to_id["<eos>"]
    )

    # 最初は文頭記号だけ。
    generated_ids = [english_bos_id]

    reached_eos = False

    # --------------------------------------------------------
    # 4. 次の英語トークンを1個ずつ生成する
    # --------------------------------------------------------

    for _ in range(max_generation_tokens):
        # 現在までに生成した英語ID列を、
        # バッチサイズ1のTensorにする。
        decoder_input_ids = torch.tensor(
            [generated_ids],
            dtype=torch.long,
            device=device,
        )

        # 91番のモデルへ、
        #
        # ・日本語文全体
        # ・現在までに生成した英語
        #
        # を入力する。
        #
        # logitsのshape:
        # [1, 現在の英語文長, 英語語彙数]
        logits = model(
            source_ids,
            decoder_input_ids,
        )

        # 最後の位置のスコアだけを取り出す。
        #
        # これは、
        # 「次にどの英語トークンを出すか」
        # に対応するスコアである。
        next_token_logits = (
            logits[0, -1].clone()
        )

        # 文生成の途中で<pad>や<bos>は不要なので、
        # 選ばれないようにスコアを-infにする。
        next_token_logits[
            english_pad_id
        ] = -float("inf")

        next_token_logits[
            english_bos_id
        ] = -float("inf")

        # 最もスコアが高い英語トークンIDを選ぶ。
        next_token_id = int(
            torch.argmax(
                next_token_logits
            ).item()
        )

        generated_ids.append(
            next_token_id
        )

        # 文末記号<eos>が出たら生成を終了する。
        if next_token_id == english_eos_id:
            reached_eos = True
            break

    # --------------------------------------------------------
    # 5. 生成されたIDを英語トークンへ戻す
    # --------------------------------------------------------

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

    translated_text = detokenize_english(
        generated_tokens
    )

    if not reached_eos:
        translated_text += (
            " [注意: 最大生成長までに"
            "<eos>が生成されませんでした]"
        )

    return (
        translated_text,
        japanese_tokens,
        generated_tokens,
    )


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # 1. 使用デバイスを決める
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"使用デバイス: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # 2. 91.pyを読み込む
    # --------------------------------------------------------

    problem91 = load_problem91_module(
        PROBLEM91_PATH
    )

    print(
        f"91番のコードを読み込みました: "
        f"{PROBLEM91_PATH}"
    )

    # --------------------------------------------------------
    # 3. 91番と同じ関数で語彙表を読み込む
    # --------------------------------------------------------

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
    # 4. 91番の学習済みモデルを読み込む
    # --------------------------------------------------------

    model = load_trained_model(
    problem91=problem91,
    device=device,
    )

    print(
        "学習済みモデルを読み込みました:",
        CHECKPOINT_PATH,
    )

    # --------------------------------------------------------
    # 5. 日本語形態素解析器を準備する
    # --------------------------------------------------------

    japanese_tagger = (
        create_mecab_tagger()
    )

    # --------------------------------------------------------
    # 6. 入力された日本語文を翻訳する
    # --------------------------------------------------------

    print(
        "\n翻訳したい日本語文を"
        "入力してください。"
    )

    print(
        "終了するときはexitと"
        "入力してください。"
    )

    while True:
        japanese_text = input(
            "\n日本語> "
        ).strip()

        if japanese_text.lower() in {
            "exit",
            "quit",
            "終了",
        }:
            print("終了します。")
            break

        if not japanese_text:
            print(
                "日本語文を入力してください。"
            )
            continue

        try:
            (
                translated_text,
                japanese_tokens,
                generated_tokens,
            ) = translate_greedily(
                model=model,
                japanese_text=(
                    japanese_text
                ),
                japanese_tagger=(
                    japanese_tagger
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
                device=device,
                max_generation_tokens=(
                    MAX_GENERATION_TOKENS
                ),
            )

            print(
                "日本語トークン:",
                " ".join(
                    japanese_tokens
                ),
            )

            print(
                "生成トークン:",
                " ".join(
                    generated_tokens
                ),
            )

            print(
                "英訳:",
                translated_text,
            )

        except (
            ValueError,
            RuntimeError,
        ) as error:
            print(
                f"翻訳できませんでした: "
                f"{error}"
            )


if __name__ == "__main__":
    main()

#出力
# 使用デバイス: cuda
# GPU: NVIDIA RTX A6000
# 91番のコードを読み込みました
# 日本語語彙数: 119,002
# 英語語彙数: 215,627
# 読み込んだモデルのエポック: 8
# そのモデルのvalid loss: 3.0717
# 学習済みモデルを読み込みました

# 翻訳したい日本語文を入力してください。
# 終了するときはexitと入力してください。

# 日本語> 春には桜が咲きます。
# 日本語トークン: 春 に は 桜 が 咲き ます 。
# 生成トークン: Cherry blossoms bloom in spring .
# 英訳: Cherry blossoms bloom in spring.

# 日本語> 翻訳テスト
# 日本語トークン: 翻訳 テスト
# 生成トークン: Translation tests
# 英訳: Translation tests

# 日本語> 京都には多くの古い寺や神社があり、毎年多くの観光客が訪れます。
# 日本語トークン: 京都 に は 多く の 古い 寺 や 神社 が あり 、 毎年 多く の 観光 客 が 訪れ ます 。
# 生成トークン: There are many old temples and shrines in Kyoto , and many tourists visit the temple every year .
# 英訳: There are many old temples and shrines in Kyoto, and many tourists visit the temple every year.

# 日本語> exit
# 終了します。