import MeCab

# 解析対象の文章
text = """
メロスは激怒した。
必ず、かの邪智暴虐の王を除かなければならぬと決意した。
メロスには政治がわからぬ。
メロスは、村の牧人である。
笛を吹き、羊と遊んで暮して来た。
けれども邪悪に対しては、人一倍に敏感であった。
"""

# MeCabのTaggerを作成する
# Taggerは文章を形態素解析するためのオブジェクト
tagger = MeCab.Tagger()

# textを形態素解析する
# parse()の結果は1単語につき1行の文字列として返ってくる
parsed_text = tagger.parse(text)

# 解析結果を1行ずつ処理する
# splitlinesは改行で分割し、リストで返す
for line in parsed_text.splitlines():
    # MeCabの解析結果の最後には"EOS"という行が出る
    # EOSは「解析終了」を表すだけなので、処理しない
    if line == "EOS":
        continue

    # MeCabの各行は「表層系\t品詞などの情報」という形
    surface, feature = line.split("\t")

    # featureには、品詞・活用形・原形などがカンマ区切りで入っている
    # それをカンマ区切りでリストに直す
    features = feature.split(",")

    # features[0]に品詞情報が入っている
    pos = features[0]

    # 品詞が動詞の単語だけを表示する
    if pos == "動詞":
        print(surface)

# 出力
# し
# 除か
# なら
# し
# わから
# 吹き
# 遊ん
# 暮し
# 来
