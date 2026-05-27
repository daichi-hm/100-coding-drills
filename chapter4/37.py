import pandas as pd
import re
import MeCab
from collections import Counter

# Wikipedia記事データを読み込む
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

# 名詞の出現頻度を数えるCounter
noun_counter = Counter()

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

        # タブがない行はスキップする
        if "\t" not in line:
            continue

        # 表層形と品詞情報に分ける
        surface, feature = line.split("\t")

        # 品詞情報をカンマで分割する
        features = feature.split(",")

        # features[0] が品詞
        pos = features[0]

        # 名詞だけを数える
        if pos == "名詞":

            # 空文字は除外する
            if surface.strip() == "":
                continue

            # 名詞の出現回数を1増やす
            noun_counter[surface] += 1

# 出現頻度の高い名詞20語を表示する
for noun, count in noun_counter.most_common(20):
    print(noun, count)

# 出力
# | 28853
# 年 26049
# 月 11362
# = 11244
# . 10576
# * 9653
# 日 8965
# 人 8759
# - 8415
# : 7556
# 1 7314
# 国 6740
# || 5881
# , 5713
# 語 5197
# 2 5149
# 的 4847
# こと 4354
# 3 4086
# ( 3725
