from __future__ import annotations

import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# 1. ファイルの場所
# ============================================================

# 91.pyが置かれているディレクトリ
BASE_DIR = Path(__file__).resolve().parent


# 90番で作ったデータの場所
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VOCAB_DIR = DATA_DIR / "vocab"


# 学習済みモデルを保存する場所
MODEL_DIR = BASE_DIR / "models"


# 訓練データ
TRAIN_JA_PATH = PROCESSED_DIR / "train.ja"
TRAIN_EN_PATH = PROCESSED_DIR / "train.en"


# 開発データ
DEV_JA_PATH = PROCESSED_DIR / "dev.ja"
DEV_EN_PATH = PROCESSED_DIR / "dev.en"


# 90番で作った語彙表
JA_VOCAB_PATH = VOCAB_DIR / "ja_vocab.json"
EN_VOCAB_PATH = VOCAB_DIR / "en_vocab.json"


# モデルや学習履歴の保存先
BEST_MODEL_PATH = MODEL_DIR / "91_best_model.pt"
LAST_MODEL_PATH = MODEL_DIR / "91_last_model.pt"
HISTORY_PATH = MODEL_DIR / "91_history.json"


# ============================================================
# 2. 学習の設定（ハイパーパラメータ）
# ============================================================

# 1回のパラメータ更新に使用する文対数
BATCH_SIZE = 16


# 訓練データ全体を何周するか
NUM_EPOCHS = 10


# 1回の更新でパラメータをどの程度動かすか
LEARNING_RATE = 1e-4


# Transformer内部で、
# 各トークンを何次元のベクトルとして表すか
D_MODEL = 256


# Multi-Head Attentionのヘッド数
# D_MODELはNHEADで割り切れる必要がある。
NHEAD = 8


# Encoderを何層重ねるか
NUM_ENCODER_LAYERS = 4


# Decoderを何層重ねるか
NUM_DECODER_LAYERS = 4


# Transformer内部のFeed Forward Networkの次元
DIM_FEEDFORWARD = 1024

# 過学習を防ぐため、
# 学習中に一部の値を0にする割合
DROPOUT = 0.1

# 位置エンコーディングが扱える最大の文長
MAX_POSITION_LENGTH = 2048


# Noneの場合は、90番で作った全ての文対を使う。
# 長すぎる文が原因でGPUメモリ不足になった場合、
# たとえば256などに変更できる。
MAX_SENTENCE_LENGTH: int | None = 256

# 勾配が極端に大きくなることを防ぐための上限
GRADIENT_CLIP_NORM = 1.0

# DataLoaderがデータを準備する並列プロセス数
NUM_WORKERS = 2

# 実行ごとの差を小さくするための乱数シード
RANDOM_SEED = 42

# ============================================================
# 3. 乱数を固定する
# ============================================================

def set_random_seed(seed: int) -> None:
    """
    実行ごとの差を小さくするため、
    PythonとPyTorchの乱数の初期値を固定する。
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. 語彙表を読み込む
# ============================================================

def load_vocabulary(
    vocabulary_path: Path,
) -> dict[str, int]:
    """
    90番で保存したJSON語彙表を読み込む。

    戻り値:
        トークンをキー、
        整数IDを値とする辞書
    """

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"語彙表が見つかりません: "
            f"{vocabulary_path}\n"
            "先に90.pyを実行してください。"
        )

    with vocabulary_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        # 90番では語彙表そのものをJSONに保存したため、
        # json.load()の結果がそのままtoken_to_idになる
        token_to_id = json.load(file)

    # 必要な特殊トークンが入っているか確認する
    required_tokens = [
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ]

    for token in required_tokens:
        if token not in token_to_id:
            raise ValueError(
                f"{vocabulary_path.name}に"
                f"特殊トークン{token}がありません。"
            )

    return token_to_id


# ============================================================
# 5. 翻訳データセット
# ============================================================

class TranslationDataset(Dataset):
    """
    トークン化済みの日英文対を読み込むDataset。

    __getitem__()が呼ばれるたびに、
    日本語と英語を整数ID列へ変換する。
    """

    def __init__(
        self,
        japanese_path: Path,
        english_path: Path,
        japanese_token_to_id: dict[str, int],
        english_token_to_id: dict[str, int],
        max_sentence_length: int | None = None,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------
        # ファイルが存在するか確認する
        # ----------------------------------------------------

        if not japanese_path.exists():
            raise FileNotFoundError(
                f"日本語データが見つかりません: "
                f"{japanese_path}"
            )

        if not english_path.exists():
            raise FileNotFoundError(
                f"英語データが見つかりません: "
                f"{english_path}"
            )

        # ----------------------------------------------------
        # ファイルを1行ずつのリストとして読み込む
        # ----------------------------------------------------

        # splitlines()は、行末の改行を除去して、
        # 各行をリストの要素にする。
        with japanese_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            japanese_lines = file.read().splitlines()

        with english_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            english_lines = file.read().splitlines()

        # 日英の行数が異なる場合は、
        # 対訳関係を保証できないので停止する
        if len(japanese_lines) != len(english_lines):
            raise ValueError(
                "トークン化済みの日英データで"
                "行数が異なります。\n"
                f"日本語: {len(japanese_lines):,}\n"
                f"英語: {len(english_lines):,}"
            )

        # ----------------------------------------------------
        # 語彙表と特殊トークンIDを保存する
        # ----------------------------------------------------

        self.japanese_token_to_id = (
            japanese_token_to_id
        )

        self.english_token_to_id = (
            english_token_to_id
        )

        self.japanese_unk_id = (
            japanese_token_to_id["<unk>"]
        )

        self.english_unk_id = (
            english_token_to_id["<unk>"]
        )

        self.japanese_bos_id = (
            japanese_token_to_id["<bos>"]
        )

        self.japanese_eos_id = (
            japanese_token_to_id["<eos>"]
        )

        self.english_bos_id = (
            english_token_to_id["<bos>"]
        )

        self.english_eos_id = (
            english_token_to_id["<eos>"]
        )

        # ----------------------------------------------------
        # 日英文対を保存する
        # ----------------------------------------------------

        self.sentence_pairs: list[
            tuple[str, str]
        ] = []

        skipped_long_pairs = 0

        for japanese_line, english_line in zip(
            japanese_lines,
            english_lines,
        ):
            # 90番で空文は除外済みだが、
            # 念のため確認する
            if (
                not japanese_line.strip()
                or not english_line.strip()
            ):
                raise ValueError(
                    "processedデータに空文があります。"
                    "90.pyの出力を確認してください。"
                )

            # MAX_SENTENCE_LENGTHを指定した場合だけ、
            # 長すぎる文対を除外する
            if max_sentence_length is not None:

                # 後で<bos>と<eos>を加えるため+2する
                japanese_length = (
                    len(japanese_line.split()) + 2
                )

                english_length = (
                    len(english_line.split()) + 2
                )

                if (
                    japanese_length
                    > max_sentence_length
                    or english_length
                    > max_sentence_length
                ):
                    skipped_long_pairs += 1
                    continue

            self.sentence_pairs.append(
                (
                    japanese_line,
                    english_line,
                )
            )

        print(
            f"[{japanese_path.stem}] "
            f"使用する文対="
            f"{len(self.sentence_pairs):,}, "
            f"長文による除外="
            f"{skipped_long_pairs:,}"
        )

    def __len__(self) -> int:
        """
        Datasetに含まれる文対数を返す。
        """

        return len(self.sentence_pairs)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        index番目の日英文対を、
        整数IDのTensorとして返す。
        """

        japanese_line, english_line = (
            self.sentence_pairs[index]
        )

        # 既に90番でトークン化済みなので、
        # 空白で分割すればトークン列になる
        japanese_tokens = japanese_line.split()
        english_tokens = english_line.split()

        # ----------------------------------------------------
        # 日本語を整数IDへ変換する
        # ----------------------------------------------------

        japanese_ids = [
            self.japanese_token_to_id.get(
                token,
                self.japanese_unk_id,
            )
            for token in japanese_tokens
        ]

        # ----------------------------------------------------
        # 英語を整数IDへ変換する
        # ----------------------------------------------------

        english_ids = [
            self.english_token_to_id.get(
                token,
                self.english_unk_id,
            )
            for token in english_tokens
        ]

        # ----------------------------------------------------
        # 文の先頭と末尾に特殊トークンを追加する
        # ----------------------------------------------------

        japanese_ids = (
            [self.japanese_bos_id]
            + japanese_ids
            + [self.japanese_eos_id]
        )

        english_ids = (
            [self.english_bos_id]
            + english_ids
            + [self.english_eos_id]
        )

        # nn.Embeddingへ入力するIDは、
        # torch.long型である必要がある
        japanese_tensor = torch.tensor(
            japanese_ids,
            dtype=torch.long,
        )

        english_tensor = torch.tensor(
            english_ids,
            dtype=torch.long,
        )

        return japanese_tensor, english_tensor


# ============================================================
# 6. ミニバッチ内の文長をそろえる
# ============================================================

def collate_translation_batch(
    batch: list[
        tuple[torch.Tensor, torch.Tensor]
    ],
    japanese_pad_id: int,
    english_pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    長さの異なる複数の文を<pad>で埋め、
    バッチとして1つのTensorにまとめる。

    戻り値のshape:
        日本語:
        [バッチサイズ, 日本語の最大文長]

        英語:
        [バッチサイズ, 英語の最大文長]
    """

    # batchの各要素は、
    #
    # (
    #     日本語IDのTensor,
    #     英語IDのTensor,
    # )
    #
    # という形になっている

    japanese_sequences = [
        pair[0]
        for pair in batch
    ]

    english_sequences = [
        pair[1]
        for pair in batch
    ]

    # バッチ内の最長文に合わせて、
    # 短い日本語文の末尾を<pad>で埋める
    japanese_batch = pad_sequence(
        japanese_sequences,
        batch_first=True,
        padding_value=japanese_pad_id,
    )

    # 英語側も同様にPaddingする
    english_batch = pad_sequence(
        english_sequences,
        batch_first=True,
        padding_value=english_pad_id,
    )

    return japanese_batch, english_batch


# ============================================================
# 7. 位置エンコーディング
# ============================================================

class PositionalEncoding(nn.Module):
    """
    各トークンが文の何番目にあるかを表す
    位置情報をEmbeddingへ加える。
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # positionのshape:
        # [max_length, 1]
        #
        # 0, 1, 2, 3, ...という位置番号を作る
        position = torch.arange(
            max_length,
            dtype=torch.float,
        ).unsqueeze(1)

        # 各次元で異なるsin・cosの周期を作る
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float,
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        positional_encoding = torch.zeros(
            max_length,
            d_model,
        )

        # 偶数番目の次元にはsinを使う
        positional_encoding[:, 0::2] = torch.sin(
            position * div_term
        )

        # 奇数番目の次元にはcosを使う
        positional_encoding[:, 1::2] = torch.cos(
            position * div_term
        )

        # バッチ次元を先頭へ追加する
        #
        # 変更前:
        # [max_length, d_model]
        #
        # 変更後:
        # [1, max_length, d_model]
        positional_encoding = (
            positional_encoding.unsqueeze(0)
        )

        # 位置エンコーディングは学習する値ではない。
        #
        # ただし、
        # model.to(device)でGPUへ移動したり、
        # state_dict()でモデルと一緒に保存したりしたい。
        #
        # そのためregister_buffer()で登録する。
        self.register_buffer(
            "positional_encoding",
            positional_encoding,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        embeddingsのshape:
        [バッチサイズ, 文長, d_model]
        """

        sequence_length = embeddings.size(1)

        if (
            sequence_length
            > self.positional_encoding.size(1)
        ):
            raise ValueError(
                f"文長{sequence_length}が、"
                "位置エンコーディングの上限"
                f"{self.positional_encoding.size(1)}"
                "を超えました。"
            )

        # 現在の文長に必要な部分だけを取り出し、
        # トークンEmbeddingへ加える
        embeddings = (
            embeddings
            + self.positional_encoding[
                :,
                :sequence_length,
            ]
        )

        return self.dropout(embeddings)


# ============================================================
# 8. Transformer翻訳モデル
# ============================================================

class TranslationTransformer(nn.Module):
    """
    日本語をEncoderへ入力し、
    Decoderから英語を予測するTransformer。
    """

    def __init__(
        self,
        japanese_vocab_size: int,
        english_vocab_size: int,
        japanese_pad_id: int,
        english_pad_id: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_position_length: int,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                "d_modelはnheadで"
                "割り切れる値にしてください。"
            )

        self.d_model = d_model
        self.japanese_pad_id = japanese_pad_id
        self.english_pad_id = english_pad_id

        # ----------------------------------------------------
        # 日本語・英語のEmbedding
        # ----------------------------------------------------

        # 日本語の整数IDを、
        # d_model次元のベクトルへ変換する
        self.japanese_embedding = nn.Embedding(
            num_embeddings=japanese_vocab_size,
            embedding_dim=d_model,
            padding_idx=japanese_pad_id,
        )

        # 英語の整数IDを、
        # d_model次元のベクトルへ変換する
        self.english_embedding = nn.Embedding(
            num_embeddings=english_vocab_size,
            embedding_dim=d_model,
            padding_idx=english_pad_id,
        )

        # 日本語と英語の両方で、
        # 同じ仕組みの位置エンコーディングを使う
        self.positional_encoding = (
            PositionalEncoding(
                d_model=d_model,
                dropout=dropout,
                max_length=max_position_length,
            )
        )

        # ----------------------------------------------------
        # Transformer本体
        # ----------------------------------------------------

        # OpenNMTやfairseqではなく、
        # PyTorchのnn.Transformerを直接使用する。
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=(
                num_encoder_layers
            ),
            num_decoder_layers=(
                num_decoder_layers
            ),
            dim_feedforward=dim_feedforward,
            dropout=dropout,

            # Tensorのshapeを、
            # [batch, sequence, feature]
            # の順にする
            batch_first=True,
        )

        # Decoderが出力したd_model次元ベクトルを、
        # 英語語彙の各トークンのスコアへ変換する
        self.output_layer = nn.Linear(
            d_model,
            english_vocab_size,
        )

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """
        2次元以上のパラメータを、
        Xavierの方法で初期化する。
        """

        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def create_causal_mask(
        target_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Decoderが未来の正解トークンを見ないようにする
        Causal Maskを作る。

        True:
            Attentionで参照してはいけない位置

        False:
            参照できる位置
        """

        return torch.triu(
            torch.ones(
                target_length,
                target_length,
                dtype=torch.bool,
                device=device,
            ),

            # 対角線より上をTrueにする
            diagonal=1,
        )

    def forward(
        self,
        japanese_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        引数:
            japanese_ids:
                [バッチサイズ, 日本語文長]

            decoder_input_ids:
                [バッチサイズ, 英語文長]

        戻り値:
            [バッチサイズ,
             英語文長,
             英語語彙数]
        """

        # ----------------------------------------------------
        # Padding Maskを作る
        # ----------------------------------------------------

        # <pad>の位置をTrueにする
        japanese_padding_mask = (
            japanese_ids.eq(
                self.japanese_pad_id
            )
        )

        english_padding_mask = (
            decoder_input_ids.eq(
                self.english_pad_id
            )
        )

        # ----------------------------------------------------
        # Decoder用のCausal Maskを作る
        # ----------------------------------------------------

        causal_mask = self.create_causal_mask(
            target_length=(
                decoder_input_ids.size(1)
            ),
            device=decoder_input_ids.device,
        )

        # ----------------------------------------------------
        # 整数IDをEmbeddingへ変換する
        # ----------------------------------------------------

        japanese_embeddings = (
            self.japanese_embedding(
                japanese_ids
            )
            * math.sqrt(self.d_model)
        )

        english_embeddings = (
            self.english_embedding(
                decoder_input_ids
            )
            * math.sqrt(self.d_model)
        )

        # ----------------------------------------------------
        # 位置情報を加える
        # ----------------------------------------------------

        japanese_embeddings = (
            self.positional_encoding(
                japanese_embeddings
            )
        )

        english_embeddings = (
            self.positional_encoding(
                english_embeddings
            )
        )

        # ----------------------------------------------------
        # Transformerへ入力する
        # ----------------------------------------------------

        transformer_output = self.transformer(
            # Encoderへの入力
            src=japanese_embeddings,

            # Decoderへの入力
            tgt=english_embeddings,

            # Decoderが未来を見ないためのマスク
            tgt_mask=causal_mask,

            # 日本語の<pad>をAttention対象外にする
            src_key_padding_mask=(
                japanese_padding_mask
            ),

            # Decoder入力の<pad>を対象外にする
            tgt_key_padding_mask=(
                english_padding_mask
            ),

            # DecoderがEncoderの出力を見る際も、
            # 日本語側の<pad>を対象外にする
            memory_key_padding_mask=(
                japanese_padding_mask
            ),
        )

        # 各位置について、
        # 英語語彙の全トークンのスコアを出す
        logits = self.output_layer(
            transformer_output
        )

        return logits


# ============================================================
# 9. 1エポック分の学習
# ============================================================

def train_one_epoch(
    model: TranslationTransformer,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    english_pad_id: int,
) -> float:
    """
    訓練データを1周する。

    戻り値:
        <pad>を除いた、
        1トークンあたりの平均loss
    """

    # 学習モードにする
    #
    # Dropoutなどが有効になる
    model.train()

    total_loss = 0.0
    total_target_tokens = 0

    progress_bar = tqdm(
        data_loader,
        desc="train",
        leave=False,
    )

    for japanese_ids, english_ids in progress_bar:

        # ----------------------------------------------------
        # バッチをGPUへ移動する
        # ----------------------------------------------------

        japanese_ids = japanese_ids.to(
            device,
            non_blocking=True,
        )

        english_ids = english_ids.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Decoder入力と正解ラベルを1つずらす
        # ----------------------------------------------------

        # 完全な英語文が、
        #
        # <bos> I like Kyoto . <eos>
        #
        # だった場合、
        #
        # Decoderへの入力:
        # <bos> I like Kyoto .
        #
        # 正解:
        # I like Kyoto . <eos>
        #
        # とする。

        decoder_input_ids = english_ids[:, :-1]
        expected_ids = english_ids[:, 1:]

        # 前のバッチで計算された勾配を0に戻す
        optimizer.zero_grad(
            set_to_none=True
        )

        # Transformerで英語の各トークンを予測する
        logits = model(
            japanese_ids=japanese_ids,
            decoder_input_ids=(
                decoder_input_ids
            ),
        )

        # ----------------------------------------------------
        # Cross Entropy Lossを計算する
        # ----------------------------------------------------

        # logitsの元のshape:
        # [batch, length, vocab]
        #
        # CrossEntropyLossへ渡すため:
        # [batch * length, vocab]
        #
        # に変形する。
        logits_for_loss = logits.reshape(
            -1,
            logits.size(-1),
        )

        # expected_idsも、
        # [batch * length]
        # に変形する。
        expected_ids_for_loss = (
            expected_ids.reshape(-1)
        )

        loss = criterion(
            logits_for_loss,
            expected_ids_for_loss,
        )

        # NaNや無限大になった場合は停止する
        if not torch.isfinite(loss):
            raise RuntimeError(
                "lossがNaNまたは無限大に"
                "なりました。\n"
                "バッチサイズ、学習率、"
                "入力文長を確認してください。"
            )

        # lossから各パラメータの勾配を計算する
        loss.backward()

        # 勾配が極端に大きくなるのを抑える
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM,
        )

        # 計算された勾配を使って、
        # モデルのパラメータを更新する
        optimizer.step()

        # ----------------------------------------------------
        # エポック全体のlossを集計する
        # ----------------------------------------------------

        # <pad>ではない正解トークン数を数える
        number_of_target_tokens = (
            expected_ids.ne(
                english_pad_id
            ).sum().item()
        )

        # criterionのlossは平均値なので、
        # トークン数を掛けて合計値へ戻す
        total_loss += (
            loss.item()
            * number_of_target_tokens
        )

        total_target_tokens += (
            number_of_target_tokens
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / total_target_tokens


# ============================================================
# 10. 開発データのlossを計算する
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model: TranslationTransformer,
    data_loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    english_pad_id: int,
) -> float:
    """
    モデルのパラメータを更新せず、
    開発データの平均lossを計算する。
    """

    # 評価モードにする
    #
    # Dropoutなどが無効になる
    model.eval()

    total_loss = 0.0
    total_target_tokens = 0

    progress_bar = tqdm(
        data_loader,
        desc="valid",
        leave=False,
    )

    for japanese_ids, english_ids in progress_bar:

        japanese_ids = japanese_ids.to(
            device,
            non_blocking=True,
        )

        english_ids = english_ids.to(
            device,
            non_blocking=True,
        )

        decoder_input_ids = english_ids[:, :-1]
        expected_ids = english_ids[:, 1:]

        logits = model(
            japanese_ids=japanese_ids,
            decoder_input_ids=(
                decoder_input_ids
            ),
        )

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            expected_ids.reshape(-1),
        )

        number_of_target_tokens = (
            expected_ids.ne(
                english_pad_id
            ).sum().item()
        )

        total_loss += (
            loss.item()
            * number_of_target_tokens
        )

        total_target_tokens += (
            number_of_target_tokens
        )

    return total_loss / total_target_tokens


# ============================================================
# 11. モデルを保存する
# ============================================================

def save_checkpoint(
    output_path: Path,
    model: TranslationTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    model_config: dict[str, int | float],
    japanese_vocab_size: int,
    english_vocab_size: int,
    special_token_ids: dict[str, int],
) -> None:
    """
    学習済みモデルと、
    92番以降でモデルを再現するための情報を保存する。
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        # 何エポック目まで学習したか
        "epoch": epoch,

        # モデルのパラメータ
        "model_state_dict": (
            model.state_dict()
        ),

        # Optimizerの状態
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),

        # 保存時点のloss
        "train_loss": train_loss,
        "valid_loss": valid_loss,

        # Transformerを再作成するための設定
        "model_config": model_config,

        # 語彙数
        "japanese_vocab_size": (
            japanese_vocab_size
        ),

        "english_vocab_size": (
            english_vocab_size
        ),

        # 特殊トークンの整数ID
        "special_token_ids": (
            special_token_ids
        ),
    }

    torch.save(
        checkpoint,
        output_path,
    )


# ============================================================
# 12. 学習履歴をJSONへ保存する
# ============================================================

def save_history(
    history: list[
        dict[str, float | int]
    ],
) -> None:
    """
    エポックごとのtrain lossとdev lossを、
    JSONファイルへ保存する。
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 13. メイン処理
# ============================================================

def main() -> None:
    # 乱数を固定する
    set_random_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # GPUが使えるか確認する
    # --------------------------------------------------------

    # 研究室の指示でGPU学習が必須なので、
    # GPUがない場合はCPUへ自動切替せず停止する。
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA対応GPUを利用できません。\n"
            "GPUノードに接続しているか、"
            "PyTorchのCUDA環境を"
            "確認してください。"
        )

    device = torch.device("cuda")

    print(f"使用デバイス: {device}")

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------------
    # 語彙表を読み込む
    # --------------------------------------------------------

    japanese_token_to_id = load_vocabulary(
        JA_VOCAB_PATH
    )

    english_token_to_id = load_vocabulary(
        EN_VOCAB_PATH
    )

    japanese_pad_id = (
        japanese_token_to_id["<pad>"]
    )

    english_pad_id = (
        english_token_to_id["<pad>"]
    )

    print(
        f"日本語語彙数: "
        f"{len(japanese_token_to_id):,}"
    )

    print(
        f"英語語彙数: "
        f"{len(english_token_to_id):,}"
    )

    # --------------------------------------------------------
    # Datasetを作る
    # --------------------------------------------------------

    train_dataset = TranslationDataset(
        japanese_path=TRAIN_JA_PATH,
        english_path=TRAIN_EN_PATH,
        japanese_token_to_id=(
            japanese_token_to_id
        ),
        english_token_to_id=(
            english_token_to_id
        ),
        max_sentence_length=(
            MAX_SENTENCE_LENGTH
        ),
    )

    dev_dataset = TranslationDataset(
        japanese_path=DEV_JA_PATH,
        english_path=DEV_EN_PATH,
        japanese_token_to_id=(
            japanese_token_to_id
        ),
        english_token_to_id=(
            english_token_to_id
        ),
        max_sentence_length=(
            MAX_SENTENCE_LENGTH
        ),
    )

    # collate_translation_batch()に、
    # 日本語・英語のpad IDをあらかじめ固定する
    collate_function = partial(
        collate_translation_batch,
        japanese_pad_id=japanese_pad_id,
        english_pad_id=english_pad_id,
    )

    # --------------------------------------------------------
    # DataLoaderを作る
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,

        # 1バッチに含める文対数
        batch_size=BATCH_SIZE,

        # 訓練データは毎エポック順番を混ぜる
        shuffle=True,

        # データ準備を行う並列プロセス数
        num_workers=NUM_WORKERS,

        # GPUへの転送を効率化する
        pin_memory=True,

        # 文長をそろえる関数
        collate_fn=collate_function,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,

        # 開発データは並び替える必要がない
        shuffle=False,

        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_function,
    )

    # --------------------------------------------------------
    # Transformerモデルを作る
    # --------------------------------------------------------

    # 92番で同じ構造を再作成できるように、
    # モデル設定を辞書として保存しておく
    model_config: dict[
        str,
        int | float
    ] = {
        "d_model": D_MODEL,
        "nhead": NHEAD,
        "num_encoder_layers": (
            NUM_ENCODER_LAYERS
        ),
        "num_decoder_layers": (
            NUM_DECODER_LAYERS
        ),
        "dim_feedforward": (
            DIM_FEEDFORWARD
        ),
        "dropout": DROPOUT,
        "max_position_length": (
            MAX_POSITION_LENGTH
        ),
    }

    model = TranslationTransformer(
        japanese_vocab_size=(
            len(japanese_token_to_id)
        ),
        english_vocab_size=(
            len(english_token_to_id)
        ),
        japanese_pad_id=japanese_pad_id,
        english_pad_id=english_pad_id,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=(
            NUM_ENCODER_LAYERS
        ),
        num_decoder_layers=(
            NUM_DECODER_LAYERS
        ),
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_position_length=(
            MAX_POSITION_LENGTH
        ),
    )

    # モデルをGPUへ移動する
    model = model.to(device)

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"モデルのパラメータ数: "
        f"{number_of_parameters:,}"
    )

    # --------------------------------------------------------
    # loss関数とOptimizerを作る
    # --------------------------------------------------------

    # <pad>の部分は正解として評価しない
    criterion = nn.CrossEntropyLoss(
        ignore_index=english_pad_id,
    )

    # 91番では基本となるAdamを使用する。
    #
    # Adam、AdamW、RAdamなどの比較は、
    # 97番で行う。
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    special_token_ids = {
        "japanese_pad_id": (
            japanese_pad_id
        ),
        "japanese_unk_id": (
            japanese_token_to_id["<unk>"]
        ),
        "japanese_bos_id": (
            japanese_token_to_id["<bos>"]
        ),
        "japanese_eos_id": (
            japanese_token_to_id["<eos>"]
        ),
        "english_pad_id": (
            english_pad_id
        ),
        "english_unk_id": (
            english_token_to_id["<unk>"]
        ),
        "english_bos_id": (
            english_token_to_id["<bos>"]
        ),
        "english_eos_id": (
            english_token_to_id["<eos>"]
        ),
    }

    # --------------------------------------------------------
    # 学習ループ
    # --------------------------------------------------------

    # 最小化したいので、
    # 最初は無限大としておく
    best_valid_loss = float("inf")

    history: list[
        dict[str, float | int]
    ] = []

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        epoch_start_time = time.time()

        # 訓練データでモデルを更新する
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            english_pad_id=english_pad_id,
        )

        # 開発データでは更新せずlossを計算する
        valid_loss = evaluate_loss(
            model=model,
            data_loader=dev_loader,
            criterion=criterion,
            device=device,
            english_pad_id=english_pad_id,
        )

        elapsed_seconds = (
            time.time()
            - epoch_start_time
        )

        print(
            f"Epoch {epoch:02d}/"
            f"{NUM_EPOCHS} | "
            f"train loss="
            f"{train_loss:.4f} | "
            f"dev loss="
            f"{valid_loss:.4f} | "
            f"time="
            f"{elapsed_seconds:.1f}秒"
        )

        # 学習履歴を追加する
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "elapsed_seconds": (
                    elapsed_seconds
                ),
            }
        )

        # 途中で停止しても履歴が残るように、
        # 毎エポック保存する
        save_history(history)

        # 毎エポック、最新モデルを保存する
        save_checkpoint(
            output_path=LAST_MODEL_PATH,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            model_config=model_config,
            japanese_vocab_size=(
                len(japanese_token_to_id)
            ),
            english_vocab_size=(
                len(english_token_to_id)
            ),
            special_token_ids=(
                special_token_ids
            ),
        )

        # 開発データのlossが、
        # これまでで最小の場合
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            # 92番以降で使うbestモデルとして保存する
            save_checkpoint(
                output_path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                model_config=model_config,
                japanese_vocab_size=(
                    len(japanese_token_to_id)
                ),
                english_vocab_size=(
                    len(english_token_to_id)
                ),
                special_token_ids=(
                    special_token_ids
                ),
            )

            print(
                "  -> bestモデルを保存しました: "
                f"{BEST_MODEL_PATH}"
            )

    print("\n学習が完了しました。")

    print(
        f"bestモデル: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"最新モデル: "
        f"{LAST_MODEL_PATH}"
    )

    print(
        f"学習履歴: "
        f"{HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()

#出力
# 使用デバイス: cuda
# GPU: NVIDIA RTX A6000
# 日本語語彙数: 119,002
# 英語語彙数: 215,627
# [train] 使用する文対=440,267, 長文による除外=19
# [dev] 使用する文対=1,166, 長文による除外=0
# モデルのパラメータ数: 148,454,987
                                                               
# Epoch 02/10 | train loss=3.6996 | dev loss=3.6066 | time=1980.8秒                                             
# Epoch 03/10 | train loss=3.1377 | dev loss=3.2792 | time=1983.6秒                                             
# Epoch 04/10 | train loss=2.8253 | dev loss=3.1587 | time=1972.4秒                                             
# Epoch 05/10 | train loss=2.6313 | dev loss=3.1059 | time=1983.8秒                                             
# Epoch 06/10 | train loss=2.4877 | dev loss=3.0772 | time=2722.1秒                                             
# Epoch 07/10 | train loss=2.3728 | dev loss=3.0883 | time=2128.8秒                                             
# Epoch 08/10 | train loss=2.2768 | dev loss=3.0717 | time=1975.8秒                                             
# Epoch 09/10 | train loss=2.1971 | dev loss=3.1107 | time=1974.3秒                                             
# Epoch 10/10 | train loss=2.1292 | dev loss=3.1204 | time=1981.5秒  