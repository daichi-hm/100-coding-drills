from pathlib import Path

import torch
from torch import nn

# =========================
# パス設定
# =========================

# このファイルが置かれているディレクトリ
BASE_DIR = Path(__file__).resolve().parent

# 70番で作成した単語埋め込みデータ
# 例: nlp100/chapter8/data/embeddings.pt
EMBEDDING_FILE = BASE_DIR / "data" / "embeddings.pt"


class BoWClassifier(nn.Module):
    """
    Bag of Words によるポジネガ分類モデル。

    処理の流れ:
    1. トークンID列を単語ベクトル列に変換する
    2. 単語ベクトルの平均を取って文ベクトルを作る
    3. 文ベクトルを線形層に入れて、2クラスのスコアを出す
    """

    def __init__(self, embedding_matrix, num_classes=2):
        """
        モデルの部品を定義する。

        embedding_matrix:
            70番で作成した単語埋め込み行列。
            形は [語彙数, 埋め込み次元数]。

        num_classes:
            分類するクラス数。
            今回はネガティブ/ポジティブの2クラスなので2。
        """

        # nn.Moduleの初期化処理を呼ぶ
        super().__init__()

        # 70番で作った単語埋め込み行列を使って、
        # 単語IDから単語ベクトルを取り出す層を作る。
        #
        # freeze=True にすると、学習時に単語埋め込み行列を更新しない。
        #
        # padding_idx=0 は、ID 0 を <PAD> として扱う指定。
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embedding_matrix,
            freeze=True,
            padding_idx=0,
        )

        # 単語ベクトルの次元数を取得する。
        # GoogleNewsの単語ベクトルなら300になる。
        embedding_dim = embedding_matrix.size(1)

        # 文ベクトルを2クラスのスコアに変換する線形層。
        #
        # 入力:  embedding_dim次元の文ベクトル
        # 出力:  num_classes次元のスコア
        #
        # 今回は num_classes=2 なので、
        # [ネガティブのスコア, ポジティブのスコア] を出す。
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, lengths):
        """
        モデルの順伝播を定義する。

        input_ids:
            パディング済みのトークンID列。
            形は [バッチサイズ, 文長]。

        lengths:
            各文のパディング前の長さ。
            形は [バッチサイズ]。

        戻り値:
            各文に対する2クラスのスコア。
            形は [バッチサイズ, 2]。
        """

        # input_idsを単語ベクトルに変換する。
        #
        # input_ids の形:
        #   [バッチサイズ, 文長]
        #
        # embeddings の形:
        #   [バッチサイズ, 文長, 埋め込み次元数]
        embeddings = self.embedding(input_ids)

        # 文中の単語ベクトルを足し合わせる。
        #
        # dim=1 は「文長方向に足す」という意味。
        #
        # embeddings の形:
        #   [バッチサイズ, 文長, 埋め込み次元数]
        #
        # summed_vectors の形:
        #   [バッチサイズ, 埋め込み次元数]
        summed_vectors = embeddings.sum(dim=1)

        # lengthsを割り算に使える形に変換する。
        #
        # lengths の元の形:
        #   [バッチサイズ]
        #
        # unsqueeze(1)後の形:
        #   [バッチサイズ, 1]
        #
        # こうしておくと、
        # [バッチサイズ, 埋め込み次元数] の summed_vectors を
        # 各文の長さで割れる。
        lengths = lengths.clamp(min=1).unsqueeze(1).to(summed_vectors.dtype)

        # 単語ベクトルの平均を取って文ベクトルにする。
        #
        # sentence_vectors の形:
        #   [バッチサイズ, 埋め込み次元数]
        sentence_vectors = summed_vectors / lengths

        # 文ベクトルを分類器に入れて、2クラスのスコアを出す。
        #
        # logits の形:
        #   [バッチサイズ, 2]
        logits = self.linear(sentence_vectors)

        return logits


def main():
    print("Bag of Wordsモデルを構築します。")

    # =========================
    # 70番で保存した単語埋め込み行列を読み込む
    # =========================

    data = torch.load(EMBEDDING_FILE, map_location="cpu")

    embedding_matrix = data["embedding_matrix"]

    print(f"embedding_matrix の形: {embedding_matrix.shape}")

    # =========================
    # モデルを作成する
    # =========================

    model = BoWClassifier(embedding_matrix)

    print(model)


if __name__ == "__main__":
    main()

# 出力
# Bag of Wordsモデルを構築します。
# embedding_matrix の形: torch.Size([300001, 300])
# BoWClassifier(
#   (embedding): Embedding(300001, 300, padding_idx=0)
#   (linear): Linear(in_features=300, out_features=2, bias=True)
# )
