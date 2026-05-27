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


# 文節chunkの中に、指定した表層系targetが含まれているか調べる関数
def chunk_has_surface(tree, chunk, target):

    # 文節に含まれる形態素を1つずる見る
    for i in range(chunk.token_pos, chunk.token_pos + chunk.token_size):

        # i番目の形態素を取得する
        token = tree.token(i)

        # 表層系がtargetと一致するれ場true
        if token.surface == target:
            return True

    # 見つからなければFalse
    return False


# 文節chunkの中に指定した助詞が含まれているか調べる関数
# particlesには探したい助詞をリストで渡す
def chunk_has_particle(tree, chunk, particles):
    # 文節に含まれる気鋭対ソを1つずつ見る
    for i in range(chunk.token_pos, chunk.token_pos + chunk.token_size):

        # i番目の形態素を取得する
        token = tree.token(i)

        # 品詞情報を取得する
        features = token.feature.split(",")

        # features[0]は品詞
        pos = features[0]

        # 品詞が助詞で、表層系が指定した助詞リストに含まれていればTrue
        if pos == "助詞" and token.surface in particles:
            return True

    # 条件に合う助詞がなければFalse
    return False


# tree.chunk_size()は、解析結果に含まれる文節の数を返す
# 文節を1つずつ見ていく
for i in range(tree.chunk_size()):

    # i番目の文節を取得する
    src_chunk = tree.chunk(i)

    # 文節にメロスが含まれているか確認
    has_melos = chunk_has_surface(tree, src_chunk, "メロス")

    # 文節に主語を表しやすい「は」または「が」が含まれているか確認
    has_subject_particle = chunk_has_particle(tree, src_chunk, ["は", "が"])

    # 「メロス」を含み、かつ「は」または「が」を含む文節だけを対象にする
    if not (has_melos and has_subject_particle):
        continue

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

# 出力結果
# メロスは        激怒した
# メロスには      わからぬ
# メロスは        牧人である
