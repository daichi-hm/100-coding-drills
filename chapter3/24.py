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
# \sは任意の空白文字を意味する
# .は任意の1文字
# +は一回以上の繰り返し
# ?は非貪欲マッチ
pattern = r"\[\[ファイル:(.+?)\|"
result = "\n".join(re.findall(pattern, uk_text))

print(result)

# 出力Royal Coat of Arms of the United Kingdom.svg
# Descriptio Prime Tabulae Europae.jpg
# Lenepveu, Jeanne d'Arc au siège d'Orléans.jpg
# London.bankofengland.arp.jpg
# Battle of Waterloo 1815.PNG
# Uk topo en.jpg
# BenNevis2005.jpg
# Population density UK 2011 census.png
# 2019 Greenwich Peninsula & Canary Wharf.jpg
# Birmingham Skyline from Edgbaston Cricket Ground crop.jpg
# Leeds CBD at night.jpg
# Glasgow and the Clyde from the air (geograph 4665720).jpg
# Palace of Westminster, London - Feb 2007.jpg
# Scotland Parliament Holyrood.jpg
# Donald Trump and Theresa May (33998675310) (cropped).jpg
# Soldiers Trooping the Colour, 16th June 2007.jpg
# City of London skyline from London City Hall - Oct 2008.jpg
# Oil platform in the North SeaPros.jpg
# Eurostar at St Pancras Jan 2008.jpg
# Heathrow Terminal 5C Iwelumo-1.jpg
# Airbus A380-841 G-XLEB British Airways (10424102995).jpg
# UKpop.svg
# Anglospeak.svg
# Royal Aberdeen Children's Hospital.jpg
# CHANDOS3.jpg
# The Fabs.JPG
# Wembley Stadium, illuminated.jpg
