import pandas as pd
import re
import MeCab
from collections import Counter

# Wikipedia記事データを読み込む
# lines=True は「1行に1つのJSONが入っている形式」を読むための指定
df = pd.read_json("jawiki-country.json.gz", lines=True)


def remove_wiki_markup(text):
    """
    Wikipedia記事本文から代表的なマークアップを簡単に除去する関数。
    完全な除去ではなく、頻度分析の邪魔になりやすい記法をざっくり消す。
    """

    # 強調マークアップを除去
    text = re.sub(r"'{2,5}", "", text)

    # 内部リンクを除去
    # [[日本]] → 日本
    # [[日本|にほん]] → にほん
    text = re.sub(r"\[\[(?:[^|\]]*?\|)?([^|\]]+?)\]\]", r"\1", text)

    # 外部リンクを除去
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)

    # HTMLタグを除去
    text = re.sub(r"<[^>]+>", "", text)

    # テンプレートを簡易的に除去
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

    # 見出しマークアップを除去
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1", text)

    return text


# MeCabのTaggerを作成する
tagger = MeCab.Tagger()

# 単語の出現頻度を数えるCounter
word_counter = Counter()

# 各記事本文を1つずつ処理する
for article_text in df["text"]:

    # Wikipediaマークアップを除去する
    clean_text = remove_wiki_markup(article_text)

    # MeCabで形態素解析する
    parsed_text = tagger.parse(clean_text)

    # MeCabの解析結果を1行ずつ処理する
    for line in parsed_text.splitlines():

        # EOSは文末を表すだけなのでスキップする
        if line == "EOS":
            continue

        # MeCabの出力は「表層形 \t 品詞情報」なので、タブがない行はスキップする
        if "\t" not in line:
            continue

        # 表層形と品詞情報に分ける
        surface, feature = line.split("\t")

        # 品詞情報をカンマで分割する
        features = feature.split(",")

        # features[0] が品詞
        pos = features[0]

        # 記号は除外する
        # 36番は単語頻度なので、「、」「。」などが上位に来るのを避ける
        if pos == "記号":
            continue

        # 空文字は除外する
        if surface.strip() == "":
            continue

        # 表層形を単語として数える
        word_counter[surface] += 1

# 出現頻度の高い20語を表示する
for word, count in word_counter.most_common(20):
    print(word, count)

# 出力
# の 84733
# は 49152
# に 46920
# が 41638
# を 36449
# た 33666
# で 32036
# | 28853
# と 26125
# 年 26049
# て 24296
# し 24042
# れ 13286
# いる 12607
# さ 12477
# 月 11362
# = 11244
# ある 11148
# . 10576
# も 9892
