import CaboCha

# 解析対象の文章
text = "メロスは激怒した。"

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


# CaboChaの元の解析結果も表示する
# parseToString() は、CaboChaで解析した結果を、人間が読める文字列として返してくれるメソッド
print(parser.parseToString(text))

print("係り受け関係:")

# 文節を1つずつ見る
for i in range(tree.chunk_size()):
    # i番目の文節を取得する
    chunk = tree.chunk(i)

    # chunk.linkはこの文節の係先の文節番号
    dst = chunk.link

    # 係り先がない場合はスキップする
    if dst == -1:
        continue

    # 係り元の文節を文字列に変換する
    src_text = chunk_to_text(tree, chunk)

    # 係り先の文節を文字列に変換する
    dst_text = chunk_to_text(tree, tree.chunk(dst))

    # 係り元 -> 係り先　の形で表示する
    print(src_text + "->" + dst_text)

# 出力結果
#   メロスは-D
#   激怒した。
# EOS

# 係り受け関係:
# メロスは->激怒した
