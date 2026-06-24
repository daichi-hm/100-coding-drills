from pathlib import Path

import torch
from gensim.models import KeyedVectors

# =========================
# 設定
# =========================

# Google Newsの学習済み単語ベクトルファイル
VECTOR_FILE = Path("../chapter6/GoogleNews-vectors-negative300.bin.gz")

# 作成した単語埋め込み行列などを保存する場所
OUTPUT_FILE = Path("data/embeddings.pt")

# 読み込む語彙数
# None にすると全単語を読み込む
MAX_VOCAB = 300000


def main():
    # 保存先ディレクトリを作成する
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("単語ベクトルの読み込みを開始します。")

    # gensimでword2vec形式のファイルを読み込む
    # binary=True は、Google Newsのファイルがバイナリ形式だから指定する
    if MAX_VOCAB is None:
        word_vectors = KeyedVectors.load_word2vec_format(
            VECTOR_FILE,
            binary=True,
        )
    else:
        word_vectors = KeyedVectors.load_word2vec_format(
            VECTOR_FILE,
            binary=True,
            limit=MAX_VOCAB,
        )

    print("単語ベクトルの読み込みが完了しました。")

    # 読み込んだ単語数
    vocab_size = len(word_vectors.index_to_key)

    # 単語ベクトルの次元数
    embedding_dim = word_vectors.vector_size

    print(f"読み込んだ語彙数: {vocab_size}")
    print(f"単語ベクトルの次元数: {embedding_dim}")

    # =========================
    # 単語埋め込み行列 E を作る
    # =========================
    #
    # 0行目は <PAD> 用に予約するので、
    # 行数は「語彙数 + 1」にする
    #
    # 例:
    # embedding_matrix[0] -> <PAD> のゼロベクトル
    # embedding_matrix[1] -> 1番目の単語のベクトル
    # embedding_matrix[2] -> 2番目の単語のベクトル
    #
    embedding_matrix = torch.zeros(
        vocab_size + 1,
        embedding_dim,
        dtype=torch.float32,
    )

    # gensimが持っている単語ベクトルを、2行目以降にコピーする
    # word_vectors.vectors の形は [vocab_size, embedding_dim]
    embedding_matrix[1:] = torch.from_numpy(word_vectors.vectors)

    # =========================
    # 単語とIDの対応表を作る
    # =========================

    # token_to_id:
    #   単語からIDを調べる辞書
    #   例: token_to_id["movie"] -> 12345
    token_to_id = {"<PAD>": 0}

    # id_to_token:
    #   IDから単語を調べるリスト
    #   例: id_to_token[12345] -> "movie"
    id_to_token = ["<PAD>"]

    # word_vectors.index_to_key には、読み込んだ単語が順番に入っている
    for i, token in enumerate(word_vectors.index_to_key, start=1):
        token_to_id[token] = i
        id_to_token.append(token)

        # =========================
    # 確認
    # =========================

    print(f"embedding_matrix の形: {embedding_matrix.shape}")
    print(f"<PAD> のID: {token_to_id['<PAD>']}")
    print(f"0行目がゼロベクトルか: {torch.all(embedding_matrix[0] == 0).item()}")

    # 例として、movie が語彙に含まれていれば確認する
    sample_word = "movie"
    if sample_word in token_to_id:
        sample_id = token_to_id[sample_word]
        print(f"{sample_word} のID: {sample_id}")
        print(f"{sample_word} のベクトル形状: {embedding_matrix[sample_id].shape}")
    else:
        print(f"{sample_word} は読み込んだ語彙に含まれていません。")

    # =========================
    # 後の問題で使えるように保存する
    # =========================

    torch.save(
        {
            "embedding_matrix": embedding_matrix,
            "token_to_id": token_to_id,
            "id_to_token": id_to_token,
        },
        OUTPUT_FILE,
    )

    print(f"保存しました: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

# 出力
# 単語ベクトルの読み込みを開始します。
# 単語ベクトルの読み込みが完了しました。
# 読み込んだ語彙数: 300000
# 単語ベクトルの次元数: 300
# embedding_matrix の形: torch.Size([300001, 300])
# <PAD> のID: 0
# 0行目がゼロベクトルか: True
# movie のID: 1122
# movie のベクトル形状: torch.Size([300])
# 保存しました: data/embeddings.pt
