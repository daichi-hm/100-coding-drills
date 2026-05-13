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

print(tree)
