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

# 形態素を保存するためのリスト
# 各要素は {"surface": 表層形, "pos": 品詞} という辞書にする
morphs = []

# 解析結果を1行ずつ処理する
# splitlinesは改行で分割し、リストで返す
for line in parsed_text.splitlines():
    # MeCabの解析結果の最後には"EOS"という行が出る
    # EOSは「解析終了」を表すだけなので、EOSが来たらループを抜ける
    if line == "EOS":
        continue

    # MeCabの各行は「表層系\t品詞などの情報」という形
    surface, feature = line.split("\t")

    # featureには、品詞・活用形・原形などがカンマ区切りで入っている
    # それをカンマ区切りでリストに直す
    features = feature.split(",")

    # features[0]は品詞情報
    pos = features[0]

    # 表層系と品詞を辞書形式でリストに保存
    morphs.append({"surface": surface, "pos": pos})

# 3個の形態素を比較するので、-2する
for i in range(len(morphs) - 2):

    # 1つ目の形態素
    first = morphs[i]
    # 2つ目の形態素
    second = morphs[i + 1]
    # 3つ目の形態素
    third = morphs[i + 2]

    # 名詞+の+名詞という形になっていれば出力する
    if first["pos"] == "名詞" and second["surface"] == "の" and third["pos"] == "名詞":

        # 条件を満たしたら、3つをつなげて表示する
        noun_phrase = first["surface"] + second["surface"] + third["surface"]
        print(noun_phrase)

# 出力
# 暴虐の王
# 村の牧人
