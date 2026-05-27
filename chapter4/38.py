import pandas as pd
import re
import MeCab
import math
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


def extract_nouns(text, tagger):
    """
    文章textをMeCabで形態素解析し、名詞だけをリストとして返す関数。
    """

    # 名詞を保存するリスト
    nouns = []

    # MeCabで形態素解析する
    parsed_text = tagger.parse(text)

    # MeCabの解析結果を1行ずつ処理する
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

        # 品詞が名詞ならリストに追加する
        if pos == "名詞":

            # 空文字は除外する
            if surface.strip() == "":
                continue

            nouns.append(surface)

    return nouns


# MeCabのTaggerを作成する
tagger = MeCab.Tagger()

# 全記事数
num_documents = len(df)

# 各名詞が何記事に出現したかを数えるCounter
# これをDF, Document Frequencyという
df_counter = Counter()

# 日本の記事に出現する名詞リストを保存する変数
japan_nouns = None

# 各記事を1つずつ処理する
# 今回はindex番号は使わないので、_ にしている
for _, row in df.iterrows():

    # 記事タイトル
    title = row["title"]

    # 記事本文
    article_text = row["text"]

    # マークアップを除去する
    clean_text = remove_wiki_markup(article_text)

    # この記事に含まれる名詞を抽出する
    nouns = extract_nouns(clean_text, tagger)

    # IDFでは「その語が何記事に出たか」を数える
    # 同じ記事内に同じ名詞が何回出ても、1記事分として数えたいのでsetを使う
    unique_nouns = set(nouns)

    # 各名詞について、出現した記事数を1増やす
    for noun in unique_nouns:
        df_counter[noun] += 1

    # タイトルが「日本」の記事なら、その名詞リストを保存しておく
    # 日本記事でTFを計算するときは、日本記事内で何回出たかが必要だから
    # set(nouns) ではなく、元の nouns を保存している
    if title == "日本":
        japan_nouns = nouns


# 念のため、日本の記事が見つからなかった場合はエラーにする
if japan_nouns is None:
    raise ValueError("title が '日本' の記事が見つかりませんでした。")


# 日本記事内での各名詞の出現回数を数える
japan_tf_count = Counter(japan_nouns)

# 日本記事内の名詞の総数
# TF = 対象単語の出現回数 / 文書内の総単語数
# の分母になる
total_nouns_in_japan_article = len(japan_nouns)

# TF-IDFの結果を保存するリスト
tfidf_scores = []

# 日本記事に出てきた名詞ごとにTF, IDF, TF-IDFを計算する
for noun, count in japan_tf_count.items():

    # TFを計算する
    # TF = 対象単語の出現回数 / 日本記事内の名詞総数
    tf = count / total_nouns_in_japan_article

    # その名詞が出現した記事数
    df_value = df_counter[noun]

    # IDFを計算する
    # IDF = log(全記事数 / その名詞が出現した記事数)
    idf = math.log(num_documents / df_value)

    # TF-IDFを計算する
    tfidf = tf * idf

    # 後で並べ替えるために、名詞・TF・IDF・TF-IDFを保存する
    tfidf_scores.append((noun, tf, idf, tfidf))


# TF-IDFスコアが高い順に並べ替える
# x[3] はタプルの4番目、つまりTF-IDF
tfidf_scores.sort(key=lambda x: x[3], reverse=True)

# 上位20語を表示する
print("名詞\tTF\tIDF\tTF-IDF")
for noun, tf, idf, tfidf in tfidf_scores[:20]:
    # 小数点以下6桁で表示
    print(f"{noun}\t{tf:.6f}\t{idf:.4f}\t{tfidf:.6f}")

# 出力
# 名詞    TF      IDF     TF-IDF
# 天皇    0.002388        2.7408  0.006546
# *「     0.001267        4.8203  0.006105
# 倭国    0.001013        4.8203  0.004884
# 列島    0.001447        3.2108  0.004648
# ::      0.001122        4.1271  0.004630
# 日本書紀        0.000796        5.5134  0.004389
# 倭      0.001013        4.1271  0.004182
# **「    0.000724        5.5134  0.003990
# 沖縄    0.001375        2.8744  0.003953
# 琉球    0.000868        4.4148  0.003834
# 県      0.005681        0.6231  0.003540
# 朝鮮    0.001267        2.6802  0.003395
# 北陸    0.000615        5.5134  0.003392
# 唐      0.001013        3.2108  0.003253
# 日本    0.025005        0.1289  0.003224
# 明治    0.001230        2.5690  0.003161
# 千島    0.000651        4.8203  0.003140
# 神武    0.000543        5.5134  0.002993
# 北海道  0.001122        2.6231  0.002943
# 台湾    0.001773        1.6014  0.002840
