import CaboCha

# 解析対象の文章
text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

# CaboChaのParserを作成
# CaboCha.Parser()は文章に対して、係り受け解析を行うためのオブジェクト
parser = CaboCha.Parser()

# treeには形態素情報と文節情報、係り受け情報が入っている
tree = parser.parse(text)


# CaboChaの文節chunkを文字列に変換する関数
# 文節に含まれる形態素をつなげて、1つの文節文字列にする
# 句読点などの記号は除く
def chunk_to_text(tree, chunk):
    # 文節を表す文字列を入れる変数
    chunk_text = ""

    # chunk.token_posは、その文節の最初の形態素番号
    # chunk.token_sizeは、その文節に含まれる形態素の個数
    for i in range(chunk.token_pos, chunk.token_pos + chunk.token_size):

        # i番目の形態素を取得する
        token = tree.token(i)

        # token.featureには品詞などの情報がカンマ区切りで入っている
        # 例:['名詞', '一般', '*', '*', '*', '*', '*']
        features = token.feature.split(",")

        # feature[0]は品詞
        pos = features[0]

        # 句読点のような「記号」は文字列に含めない
        # token.surfaceでその形態素の表層系が分かる
        if pos != "記号":
            chunk_text += token.surface

    # 完成した文字列(1つの文節)を返す
    return chunk_text


# tree.chunk_size()は、解析結果に含まれる文節の数を返す
# 文節を1つずつ見ていく
for i in range(tree.chunk_size()):

    # i番目の文節を取得する
    src_chunk = tree.chunk(i)

    # scr_chunk.linkは、その文節の係り先の文節番号を表す
    # 係り先がない場合は-1になる
    dst_index = src_chunk.link

    # 係り先がない文節は、係り受けペアを作れないのでスキップする
    if dst_index == -1:
        continue

    # 係り先の文節を取得する
    dst_chunk = tree.chunk(dst_index)

    # 係り元の文節を文字列に変換する
    src_text = chunk_to_text(tree, src_chunk)

    # 係り先の文節を文字列に変換する
    dst_text = chunk_to_text(tree, dst_chunk)

    # 係り元と係り先をタブ区切りで表示する
    print(src_text + "\t" + dst_text)

# 出力
# メロスは        激怒した
# 激怒した        決意した
# 必ず    除かなければならぬと
# かの    邪智暴虐の
# 邪智暴虐の      王を
# 王を    除かなければならぬと
# 除かなければならぬと    決意した
# 決意した        わからぬ
# メロスには      わからぬ
# 政治が  わからぬ
# わからぬ        牧人である
# メロスは        牧人である
# 村の    牧人である
# 牧人である      暮して来た
# 笛を    吹き
# 吹き    暮して来た
# 羊と    遊んで
# 遊んで  暮して来た
# 暮して来た      敏感であった
# けれども        敏感であった
# 邪悪に対しては  敏感であった
# 人一倍に        敏感であった
