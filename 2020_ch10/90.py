from __future__ import annotations

import json
import tarfile
from collections import Counter
from pathlib import Path

import MeCab
import requests
from sacremoses import MosesTokenizer
from tqdm import tqdm


# ============================================================
# 1. パスや設定
# ============================================================

# KFTTのダウンロードURL
# 実際に通信するのは、後でrequests.get()を呼び出したとき。
KFTT_URL = (
    "https://www.phontron.com/kftt/download/"
    "kftt-data-1.0.tar.gz"
)


# 90.pyが置かれているディレクトリ
BASE_DIR = Path(__file__).resolve().parent

# データ関係のファイルを保存する場所
DATA_DIR = BASE_DIR / "data"

# ダウンロードした圧縮ファイルを保存する場所
DOWNLOAD_DIR = DATA_DIR / "download"

# 圧縮ファイルを展開する場所
RAW_DIR = DATA_DIR / "raw"

# 自分でトークン化したデータを保存する場所
PROCESSED_DIR = DATA_DIR / "processed"

# 語彙表を保存する場所
VOCAB_DIR = DATA_DIR / "vocab"

# ダウンロードするtar.gzファイルの保存先
ARCHIVE_PATH = DOWNLOAD_DIR / "kftt-data-1.0.tar.gz"


# 展開後のKFTTディレクトリ
KFTT_ROOT = RAW_DIR / "kftt-data-1.0"

# 未トークン化の元データが入っているディレクトリ
ORIGINAL_DATA_DIR = KFTT_ROOT / "data" / "orig"


# KFTT内のファイル名
#
# たとえばtrainでは、
# kyoto-train.jaとkyoto-train.enを処理する。
SPLITS = {
    "train": "kyoto-train",
    "dev": "kyoto-dev",
    "test": "kyoto-test",
}


# Transformerで使用する特殊トークン
#
# <pad>:
#   バッチ内で文の長さをそろえるために使う。
#
# <unk>:
#   語彙表に存在しない未知語を表す。
#
# <bos>:
#   文の開始を表す。
#
# <eos>:
#   文の終了を表す。
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]


# ============================================================
# 2. KFTTをダウンロードする
# ============================================================

def download_kftt() -> None:
    """
    KFTTのtar.gzファイルをダウンロードする。

    既にファイルが存在する場合は、
    再ダウンロードせずに終了する。
    """

    # 既にダウンロード済みなら何もしない
    if ARCHIVE_PATH.exists():
        print(f"[skip] ダウンロード済みです: {ARCHIVE_PATH}")
        return

    # data/downloadディレクトリを作る
    #
    # parents=True:
    #   必要なら親ディレクトリも作る
    #
    # exist_ok=True:
    #   既に存在していてもエラーにしない
    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[download] {KFTT_URL}")

    # ダウンロード途中のファイルは、
    # 正式なファイルと区別するため.partという名前にする
    temporary_path = ARCHIVE_PATH.with_name(
        ARCHIVE_PATH.name + ".part"
    )

    try:
        # stream=Trueにすると、
        # ファイル全体を一度にメモリへ読み込まず、
        # 少しずつダウンロードできる
        with requests.get(
            KFTT_URL,
            stream=True,
            timeout=(10, 120),
        ) as response:

            # 404や500などのHTTPエラーがあれば、
            # ここで例外を発生させる
            response.raise_for_status()

            # ファイル全体のサイズ
            #
            # サーバーからサイズが送られてこない場合は0になる
            total_size = int(
                response.headers.get("content-length", 0)
            )

            # 一時ファイルを書き込みモードで開く
            with temporary_path.open("wb") as output_file:

                # tqdmでダウンロード進捗を表示する
                with tqdm(
                    total=total_size if total_size > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc="KFTT download",
                ) as progress_bar:

                    # 1MBずつダウンロードする
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if not chunk:
                            continue

                        output_file.write(chunk)
                        progress_bar.update(len(chunk))

        # 最後まで正常にダウンロードできた場合だけ、
        # .partファイルを正式な名前へ変更する
        temporary_path.replace(ARCHIVE_PATH)

    except requests.RequestException as error:
        # 通信に失敗した場合、
        # 途中まで作られたファイルを削除する
        temporary_path.unlink(missing_ok=True)

        raise RuntimeError(
            "KFTTのダウンロードに失敗しました。\n"
            "GPUサーバ上で実行している場合は、"
            "http_proxyとhttps_proxyも確認してください。"
        ) from error


# ============================================================
# 3. tar.gzを展開する
# ============================================================

def extract_kftt() -> None:
    """
    ダウンロードしたtar.gzファイルを展開する。

    既に展開済みの場合は何もしない。
    """

    # 元データのディレクトリが既にあれば、
    # 展開済みと判断する
    if ORIGINAL_DATA_DIR.exists():
        print(f"[skip] 展開済みです: {KFTT_ROOT}")
        return

    # data/rawディレクトリを作る
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[extract] {ARCHIVE_PATH}")

    # r:gz:
    # gzip形式で圧縮されたtarファイルを読み込む
    with tarfile.open(
        ARCHIVE_PATH,
        mode="r:gz",
    ) as archive:
        archive.extractall(RAW_DIR)


# ============================================================
# 4. 日本語・英語のトークナイザーを準備する
# ============================================================

def create_japanese_tokenizer() -> MeCab.Tagger:
    """
    日本語を形態素単位に分割するMeCabを準備する。
    """

    try:
        # -Owakatiを指定すると、
        # MeCabの結果が空白区切りの分かち書きになる
        tokenizer = MeCab.Tagger("-Owakati")

        # 環境によっては、最初に空文字列を解析すると
        # 安定して利用できる
        tokenizer.parse("")

        return tokenizer

    except RuntimeError as error:
        raise RuntimeError(
            "MeCabの初期化に失敗しました。\n"
            "MeCab本体と辞書が利用できるか確認してください。"
        ) from error


def create_english_tokenizer() -> MosesTokenizer:
    """
    英語を単語単位に分割するMosesTokenizerを準備する。
    """

    # lang="en"で英語用の規則を使用する
    return MosesTokenizer(lang="en")


# ============================================================
# 5. 1文をトークン化する
# ============================================================

def tokenize_japanese(
    text: str,
    tokenizer: MeCab.Tagger,
) -> list[str]:
    """
    日本語文をMeCabで形態素単位に分割する。
    """

    # 行末の改行や、文の前後の空白を取り除く
    text = text.strip()

    # 空文字列なら空リストを返す
    if not text:
        return []

    # MeCabで分かち書きする
    parsed_text = tokenizer.parse(text)

    # 念のため、解析結果がNoneかどうかを確認する
    if parsed_text is None:
        return []

    return parsed_text.strip().split()


def tokenize_english(
    text: str,
    tokenizer: MosesTokenizer,
) -> list[str]:
    """
    英語文をMosesTokenizerで単語単位に分割する。
    """

    # 行末の改行や、文の前後の空白を取り除く
    text = text.strip()

    # 空文字列なら空リストを返す
    if not text:
        return []

    tokens = tokenizer.tokenize(
        text,
        return_str=False,
        escape=False,
    )

    return tokens


# ============================================================
# 6. ファイルの行数を数える
# ============================================================

def count_lines(file_path: Path) -> int:
    """
    テキストファイルの行数を数える。
    """

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return sum(1 for _ in file)


# ============================================================
# 7. train・dev・testをトークン化する
# ============================================================

def preprocess_split(
    split_name: str,
    source_prefix: str,
    japanese_tokenizer: MeCab.Tagger,
    english_tokenizer: MosesTokenizer,
) -> int:
    """
    train、dev、testのいずれかをトークン化する。

    日本語または英語のどちらかが空の場合は、
    その行の日英を文対ごと除外する。

    戻り値:
        実際に保存した文対の数
    """

    #入力ファイルと出力ファイルのパスを作る

    # 元の日本語ファイル
    # 例:
    # data/raw/kftt-data-1.0/data/orig/kyoto-train.ja
    source_ja_path = (
        ORIGINAL_DATA_DIR / f"{source_prefix}.ja"
    )

    # 元の英語ファイル
    # 例:
    # data/raw/kftt-data-1.0/data/orig/kyoto-train.en
    source_en_path = (
        ORIGINAL_DATA_DIR / f"{source_prefix}.en"
    )

    # トークン化後の日本語ファイル
    # 例:
    # data/processed/train.ja
    output_ja_path = (
        PROCESSED_DIR / f"{split_name}.ja"
    )

    # トークン化後の英語ファイル
    # 例:
    # data/processed/train.en
    output_en_path = (
        PROCESSED_DIR / f"{split_name}.en"
    )

    #元データが存在するか確認する

    if not source_ja_path.exists():
        raise FileNotFoundError(
            f"日本語ファイルが見つかりません: "
            f"{source_ja_path}"
        )

    if not source_en_path.exists():
        raise FileNotFoundError(
            f"英語ファイルが見つかりません: "
            f"{source_en_path}"
        )

    #日本語と英語の行数を確認する

    ja_line_count = count_lines(source_ja_path)
    en_line_count = count_lines(source_en_path)

    print(
        f"[{split_name}] "
        f"日本語={ja_line_count:,}行, "
        f"英語={en_line_count:,}行"
    )

    # 行数そのものが異なる場合は、
    # どの日本語と英語が対応しているか保証できないため停止する
    if ja_line_count != en_line_count:
        raise ValueError(
            f"{split_name}の日本語と英語で"
            f"行数が異なります。\n"
            f"日本語: {ja_line_count}\n"
            f"英語: {en_line_count}"
        )

    #出力先のディレクトリを作る

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    #処理結果を数える変数を用意する

    # 正常に保存した文対の数
    written_pairs = 0

    # 空文のため除外した文対の数
    skipped_empty_pairs = 0

    #入力・出力ファイルを開く
    with (
        source_ja_path.open(
            "r",
            encoding="utf-8",
        ) as ja_input,

        source_en_path.open(
            "r",
            encoding="utf-8",
        ) as en_input,

        # "w"モードなので、ファイルが既にある場合は
        # 中身を消して最初から書き直す
        output_ja_path.open(
            "w",
            encoding="utf-8",
        ) as ja_output,

        output_en_path.open(
            "w",
            encoding="utf-8",
        ) as en_output,
    ):
        # 日本語と英語の同じ行番号を組み合わせる
        paired_lines = zip(
            ja_input,
            en_input,
        )

        # 進捗バーを付ける
        progress_bar = tqdm(
            paired_lines,
            total=ja_line_count,
            desc=f"tokenize {split_name}",
        )

        # 行番号を付けて、文対を1つずつ処理する
        for line_number, pair in enumerate(
            progress_bar,
            start=1,
        ):
            # pairには、
            # (日本語の1行, 英語の1行)
            # が入っている
            japanese_line, english_line = pair

            # 日本語を形態素単位に分割する
            japanese_tokens = tokenize_japanese(
                japanese_line,
                japanese_tokenizer,
            )

            # 英語を単語単位に分割する
            english_tokens = tokenize_english(
                english_line,
                english_tokenizer,
            )

            #空文を確認する
            # 日本語または英語のどちらかが空なら、
            # その行の日英を両方とも保存しない
            if not japanese_tokens or not english_tokens:
                skipped_empty_pairs += 1

                # tqdm.write()を使うと、
                # 進捗バーを大きく崩さずに警告を表示できる
                tqdm.write(
                    f"[warning] "
                    f"{split_name}の{line_number}行目を"
                    "空文のため文対ごと除外します。"
                    f" 日本語トークン数="
                    f"{len(japanese_tokens)},"
                    f" 英語トークン数="
                    f"{len(english_tokens)}"
                )

                # 日本語・英語のどちらも書き込まず、
                # 次の行へ進む
                continue

            # =================================================
            # 8. トークン化結果を保存する
            # =================================================

            # 日本語トークンを空白でつないで保存する
            ja_output.write(
                " ".join(japanese_tokens) + "\n"
            )

            # 英語トークンを空白でつないで保存する
            en_output.write(
                " ".join(english_tokens) + "\n"
            )

            # 日英を正常に保存できたので1増やす
            written_pairs += 1

    #このsplitの処理結果を表示する
    print(
        f"[{split_name}] "
        f"元の文対数={ja_line_count:,}, "
        f"保存={written_pairs:,}, "
        f"空文による除外={skipped_empty_pairs:,}"
    )

    # 念のため、計算が合っているか確認する
    if written_pairs + skipped_empty_pairs != ja_line_count:
        raise RuntimeError(
            f"{split_name}の文対数の集計が一致しません。\n"
            f"元の文対数: {ja_line_count}\n"
            f"保存数: {written_pairs}\n"
            f"除外数: {skipped_empty_pairs}"
        )

    # 実際に保存した文対数をmain()へ返す
    return written_pairs


# ============================================================
# 8. 訓練データから語彙表を作る
# ============================================================

def build_vocabulary(
    tokenized_file_path: Path,
) -> dict[str, int]:
    """
    トークン化済み訓練データから語彙表を作る。

    戻り値:
        トークンをキー、
        整数IDを値とする辞書
    """

    # 各トークンが何回出現したかを数える
    token_counter: Counter[str] = Counter()

    with tokenized_file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in tqdm(
            file,
            desc=f"build vocab {tokenized_file_path.name}",
        ):
            # トークン化済みなので、
            # 空白でsplit()すればトークン列になる
            tokens = line.strip().split()

            token_counter.update(tokens)

    # 最初に特殊トークンへIDを割り当てる
    #
    # <pad> = 0
    # <unk> = 1
    # <bos> = 2
    # <eos> = 3
    token_to_id = {
        token: token_id
        for token_id, token in enumerate(SPECIAL_TOKENS)
    }

    # 出現回数が多い順に並べる
    #
    # 出現回数が同じ場合は、
    # トークンの文字列順に並べる。
    # こうすることで、毎回同じIDを割り当てられる。
    sorted_tokens = sorted(
        token_counter.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    for token, frequency in sorted_tokens:
        # 万が一、元データに特殊トークンと同じ文字列が
        # 含まれていた場合は重複して登録しない
        if token in token_to_id:
            continue

        # 現在の語彙数を次のIDとして使う
        token_to_id[token] = len(token_to_id)

    return token_to_id


# ============================================================
# 9. 語彙表をJSONとして保存する
# ============================================================

def save_vocabulary(
    token_to_id: dict[str, int],
    output_path: Path,
) -> None:
    """語彙表をJSONファイルとして保存する。"""

    VOCAB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            token_to_id,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 10. トークン化後の例を表示する
# ============================================================

def show_samples(
    number_of_samples: int = 3,
) -> None:
    """
    トークン化済み訓練データの先頭を表示する。
    """

    train_ja_path = PROCESSED_DIR / "train.ja"
    train_en_path = PROCESSED_DIR / "train.en"

    print("\n=== トークン化後の例 ===")

    with (
        train_ja_path.open(
            "r",
            encoding="utf-8",
        ) as ja_file,

        train_en_path.open(
            "r",
            encoding="utf-8",
        ) as en_file,
    ):
        for index, (
            japanese_line,
            english_line,
        ) in enumerate(
            zip(ja_file, en_file),
            start=1,
        ):
            print(
                f"[{index}] JA: "
                f"{japanese_line.rstrip()}"
            )

            print(
                f"    EN: "
                f"{english_line.rstrip()}"
            )

            if index >= number_of_samples:
                break


# ============================================================
# 11. メイン処理
# ============================================================

def main() -> None:
    """
    90番の処理全体を実行する。
    """

    # 1. KFTTをダウンロードする
    download_kftt()

    # 2. tar.gzを展開する
    extract_kftt()

    # 3. 日本語・英語のトークナイザーを準備する
    japanese_tokenizer = create_japanese_tokenizer()
    english_tokenizer = create_english_tokenizer()

    # splitごとの文対数を保存する
    split_statistics: dict[str, int] = {}

    # 4. train、dev、testを自分でトークン化する
    for split_name, source_prefix in SPLITS.items():
        number_of_pairs = preprocess_split(
            split_name=split_name,
            source_prefix=source_prefix,
            japanese_tokenizer=japanese_tokenizer,
            english_tokenizer=english_tokenizer,
        )

        split_statistics[split_name] = number_of_pairs

    # 5. 訓練データだけを使って日本語語彙を作る
    japanese_vocabulary = build_vocabulary(
        PROCESSED_DIR / "train.ja"
    )

    # 6. 訓練データだけを使って英語語彙を作る
    english_vocabulary = build_vocabulary(
        PROCESSED_DIR / "train.en"
    )

    # 7. 語彙表を保存する
    save_vocabulary(
        japanese_vocabulary,
        VOCAB_DIR / "ja_vocab.json",
    )

    save_vocabulary(
        english_vocabulary,
        VOCAB_DIR / "en_vocab.json",
    )

    # 8. 結果を表示する
    print("\n=== 前処理結果 ===")

    for split_name, number_of_pairs in (
        split_statistics.items()
    ):
        print(
            f"{split_name}: "
            f"{number_of_pairs:,}文対"
        )

    print(
        f"日本語語彙数: "
        f"{len(japanese_vocabulary):,}"
    )

    print(
        f"英語語彙数: "
        f"{len(english_vocabulary):,}"
    )

    # 9. トークン化後の先頭3件を表示する
    show_samples(number_of_samples=3)

    print(
        f"\nトークン化済みデータ: "
        f"{PROCESSED_DIR}"
    )

    print(
        f"語彙表: "
        f"{VOCAB_DIR}"
    )


# この90.pyを直接実行したときだけmain()を呼び出す
if __name__ == "__main__":
    main()

#出力
# [train] 日本語=440,288行, 英語=440,288行
# [warning] trainの204446行目を空文のため文対ごと除外します。 日本語トークン数=89, 英語トークン数=0             
# [warning] trainの339048行目を空文のため文対ごと除外します。 日本語トークン数=15, 英語トークン数=0             
# tokenize train: 100%|███████████████████████████████████████████████| 440288/440288 [01:08<00:00, 6394.92it/s]
# [train] 元の文対数=440,288, 保存=440,286, 空文による除外=2
# [dev] 日本語=1,166行, 英語=1,166行
# tokenize dev: 100%|█████████████████████████████████████████████████████| 1166/1166 [00:00<00:00, 6985.32it/s]
# [dev] 元の文対数=1,166, 保存=1,166, 空文による除外=0
# [test] 日本語=1,160行, 英語=1,160行
# tokenize test: 100%|████████████████████████████████████████████████████| 1160/1160 [00:00<00:00, 5897.48it/s]
# [test] 元の文対数=1,160, 保存=1,160, 空文による除外=0
# build vocab train.ja: 440286it [00:02, 187422.25it/s]
# build vocab train.en: 440286it [00:02, 189749.16it/s]

# === 前処理結果 ===
# train: 440,286文対
# dev: 1,166文対
# test: 1,160文対
# 日本語語彙数: 119,002
# 英語語彙数: 215,627

# === トークン化後の例 ===
# [1] JA: 雪舟 （ せっ しゅう 、 1420 年 （ 応永 27 年 ）- 1506 年 （ 永正 3 年 ） ） は 号 で 、 15 世紀 後半 室町 時代 に 活躍 し た 水墨 画家 ・ 禅僧 で 、 画聖 と も 称え られる 。
#     EN: Known as Sesshu ( 1420 - 1506 ) , he was an ink painter and Zen monk active in the Muromachi period in the latter half of the 15th century , and was called a master painter .
# [2] JA: 日本 の 水墨 画 を 一変 さ せ た 。
#     EN: He revolutionized the Japanese ink painting .
# [3] JA: 諱 は 「 等楊 （ とう よう ） 」 、 もしくは 「 拙 宗 （ せっ しゅう ） 」 と 号し た 。
#     EN: He was given the posthumous name " Toyo " or " Sesshu ( 拙 宗 ) . "