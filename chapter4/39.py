import pandas as pd
import re
import MeCab
import matplotlib

# WSLなど画面表示が不安定な環境でも画像保存できるようにする
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
    # 例: ''日本'' → 日本
    text = re.sub(r"'{2,5}", "", text)

    # 内部リンクを除去
    # 例: [[日本]] → 日本
    # 例: [[日本|にほん]] → にほん
    text = re.sub(r"\[\[(?:[^|\]]*?\|)?([^|\]]+?)\]\]", r"\1", text)

    # 外部リンクを除去
    # 例: [http://example.com 表示名] や http://example.com
    text = re.sub(r"\[https?://[^\]]+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)

    # HTMLタグを除去
    # 例: <ref>...</ref>, <br /> など
    text = re.sub(r"<[^>]+>", "", text)

    # テンプレートを簡易的に除去
    # 例: {{...}}
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

    # 見出しマークアップを除去
    # 例: == 歴史 == → 歴史
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

    # MeCabの解析結果を1行ずつ見る
    for line in parsed_text.splitlines():

        # EOSは文末を表すだけなのでスキップする
        if line == "EOS":
            continue

        # MeCabの出力は「表層形 \t 品詞情報」なので、
        # タブがない行はスキップする
        if "\t" not in line:
            continue

        # 表層形と品詞情報に分ける
        surface, feature = line.split("\t")

        # 品詞情報をカンマで分割する
        features = feature.split(",")

        # features[0] が品詞
        pos = features[0]

        # 記号は除外する
        # 「、」「。」などが上位に来るのを避けるため
        if pos == "記号":
            continue

        # 空文字は除外する
        if surface.strip() == "":
            continue

        # 表層形を単語として数える
        word_counter[surface] += 1


# 出現頻度の高い順に単語を並べる
# most_common() は、出現頻度が高い順に (単語, 出現頻度) を返す
word_freqs = word_counter.most_common()

# 順位を作る
# 1位, 2位, 3位, ... というリスト
ranks = list(range(1, len(word_freqs) + 1))

# 出現頻度だけを取り出す
# word_freqs は [(単語, 頻度), ...] なので、freq だけを取り出す
frequencies = [freq for word, freq in word_freqs]


# 両対数グラフを描く
plt.figure(figsize=(8, 6))

# loglog は、横軸・縦軸の両方を対数スケールにする
plt.loglog(ranks, frequencies)

# グラフのタイトルと軸ラベルを設定する
# 日本語フォント問題を避けるため、英語で書いている
plt.title("Zipf's Law")
plt.xlabel("Rank")
plt.ylabel("Frequency")

# グリッドを表示する
plt.grid(True)

# 画像として保存する
output_file = "zipf_law.png"
plt.savefig(output_file, bbox_inches="tight")

print(f"{output_file} を保存しました。")

# 出力は"zipf_law.png"として保存されている
