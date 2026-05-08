import pandas as pd
import re
import requests

df = pd.read_json("jawiki-country.json.gz", lines=True)
#'title'列が'イギリス'である行を抽出
# その行の'text'列を取得<
uk_text = df.query('title=="イギリス"')["text"].values[0]

# 基礎情報のフィールド名と値を抽出
# re.DOTALLを使うと.*が改行文字を含めてマッチできる
pattern = r"^\{\{基礎情報.*?$(.*?)^\}\}"
template = re.findall(pattern, uk_text, re.MULTILINE + re.DOTALL)

# フィールド名と値を手に入れる
# パターンA(?=パターンB)は肯定先読み
# パターンAの直後にBが来る場合はマッチする(取得するのはAのみ)
pattern = r"^\|(.+?)\s*=\s*(.+?)(?:(?=\n\|)|(?=\n$))"
result = re.findall(pattern, template[0], re.MULTILINE + re.DOTALL)

# wikiでは'を使って強調している
# subはreに含まれる文字列置換用の関数
pattern = re.compile(r"\'{2,5}", re.MULTILINE + re.DOTALL)
result = {i[0]: pattern.sub("", i[1]) for i in result}

# 内部リンクマークアップを除去
# ??は非貪欲マッチ
# []の中の先頭に^をつけると、それは除外するって意味になる
# r"\1"とすることで、1番目のグループに保存したものに上書きできる
pattern = r"\[\[(?:[^|]*?\|)??([^|]*?)\]\]"
result = {k: re.sub(pattern, r"\1", v) for k, v in result.items()}

# 外部リンク除去
# httpsの直後に?があるのは、httpとhttpsの場合があるため
# \wは任意の英数字を表している
pattern = r"https?://[\w!?/\+\-_~=;\.,*&@#$%\(\)\'\[\]]+"
result = {k: re.sub(pattern, "", v) for k, v in result.items()}

# htmlタグ除去
pattern = r"<.+?>"
result = {k: re.sub(pattern, "", v) for k, v in result.items()}

# langや仮リンクの削除
pattern = r"\{\{(?:lang|仮リンク)(?:[^|]*?\|)*?([^|]*?)\}\}"
result = {k: re.sub(pattern, r"\1", v) for k, v in result.items()}

# 国旗画像のURLを取得する
# urlにスペースはないので_に置き換える
url_file = result["国旗画像"].replace(" ", "_")
# Wikimedia Commons の API に対して、画像のURLを教えてもらっている
url = (
    "https://commons.wikimedia.org/w/api.php?action=query&titles=File:"
    + url_file
    + "&prop=imageinfo&iiprop=url&format=json"
)

# headersを設定しなかったら、下記のエラーが発生したため
# Please set a user-agent and respect our robot policy https://w.wiki/4wJS. See also https://phabricator.wikimedia.org/T400119.
headers = {"User-Agent": "100KnocksBot/1.0"}

data = requests.get(url, headers=headers)


print(re.search(r'"url":"(.+?)"', data.text).group(1))

# 出力
# https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_the_United_Kingdom_%283-5%29.svg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=original
