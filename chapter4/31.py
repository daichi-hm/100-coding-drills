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

    # features[6]に原型
    base_form = features[6]

    # 品詞が動詞のものだけ、表層系と原型を表示する
    if pos == "動詞":
        print(surface, base_form)

# 出力
# し する
# 除か 除く
# なら なる
# し する
# わから わかる
# 吹き 吹く
# 遊ん 遊ぶ
# 暮し 暮す
# 来 来る
