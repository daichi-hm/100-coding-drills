import pandas as pd
import re

df = pd.read_json("jawiki-country.json.gz", lines=True)
#'title'列が'イギリス'である行を抽出
# その行の'text'列を取得<
uk_text = df.query('title=="イギリス"')["text"].values[0]

# 正規表現を使うときは基本raw文字列にする
# そうすることで、\がただの文字としてみなされる
# raw文字列にしないと\\と2つ打つ必要がある
#'\'を打つのは、[や]をエスケープしてただの文字と見るため
#'.*'は任意の文字の0回以上の繰り返しを意味する
# .*?は非貪欲マッチというもので、なるべく少なくなるようにマッチする
# ()を?:で始めると、そこの部分は抽出しない
#'?'は0回または一回
pattern = r"^.*\[\[Category:(.*?)(?:\|.*)?\]\].*$"
# re.findallは一致した文字列をリストで返す
# 改行区切りの文字列に変換
# re.MULTILINEを入れることによって、文字列の先頭だけでなく、行ごとの判定に変えられる
result = "\n".join(re.findall(pattern, uk_text, re.MULTILINE))
print(result)

# 出力結果
# イギリス
# イギリス連邦加盟国
# 英連邦王国
# G8加盟国
# 欧州連合加盟国
# 海洋国家
# 現存する君主国
# 島国
# 1801年に成立した国家・領域
