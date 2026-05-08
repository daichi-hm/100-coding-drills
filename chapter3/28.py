import pandas as pd
import re

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

result = dict(result)
for k, v in result.items():
    print(k + ":" + v)

# 出力
# 略名:イギリス
# 日本語国名:グレートブリテン及び北アイルランド連合王国
# 公式国名:United Kingdom of Great Britain and Northern Ireland英語以外での正式国名:
# *An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath（スコットランド・ゲール語）
# *Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon（ウェールズ語）
# *Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann（アイルランド語）
# *An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh（コーンウォール語）
# *Unitit Kinrick o Great Breetain an Northren Ireland（スコットランド語）
# **Claught Kängrick o Docht Brätain an Norlin Airlann、Unitet Kängdom o Great Brittain an Norlin Airlann（アルスター・スコットランド語）
# 国旗画像:Flag of the United Kingdom.svg
# 国章画像:[[ファイル:Royal Coat of Arms of the United Kingdom.svg|85px|イギリスの国章]]
# 国章リンク:（国章）
# 標語:Dieu et mon droit（フランス語:神と我が権利）
# 国歌:[[女王陛下万歳|God Save the Queen]]{{en icon}}神よ女王を護り賜え{{center|ファイル:United States Navy Band - God Save the Queen.ogg}}
# 地図画像:Europe-UK.svg
# 位置画像:United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
# 公用語:英語
# 首都:ロンドン（事実上）
# 最大都市:ロンドン
# 元首等肩書:女王
# 元首等氏名:エリザベス2世
# 首相等肩書:首相
# 首相等氏名:ボリス・ジョンソン
# 他元首等肩書1:貴族院議長
# 他元首等氏名1:ノーマン・ファウラー
# 他元首等肩書2:庶民院議長
# 他元首等氏名2:Lindsay Hoyle
# 他元首等肩書3:最高裁判所長官
# 他元首等氏名3:ブレンダ・ヘイル
# 面積順位:76
# 面積大きさ:1 E11
# 面積値:244,820
# 水面積率:1.3%
# 人口統計年:2018
# 人口順位:22
# 人口大きさ:1 E7
# 人口値:6643万5600{{Cite web|url=|title=Population estimates - Office for National Statistics|accessdate=2019-06-26|date=2019-06-26}}
# 人口密度値:271
# GDP統計年元:2012
# GDP値元:1兆5478億[>Data and Statistics>World Economic Outlook Databases>By Countrise>United Kingdom]
# GDP統計年MER:2012
# GDP順位MER:6
# GDP値MER:2兆4337億
# GDP統計年:2012
# GDP順位:6
# GDP値:2兆3162億
# GDP/人:36,727
# 建国形態:建国
# 確立形態1:イングランド王国／スコットランド王国（両国とも1707年合同法まで）
# 確立年月日1:927年／843年
# 確立形態2:グレートブリテン王国成立（1707年合同法）
# 確立年月日2:1707年{{0}}5月{{0}}1日
# 確立形態3:グレートブリテン及びアイルランド連合王国成立（1800年合同法）
# 確立年月日3:1801年{{0}}1月{{0}}1日
# 確立形態4:現在の国号「グレートブリテン及び北アイルランド連合王国」に変更
# 確立年月日4:1927年{{0}}4月12日
# 通貨:UKポンド (£)
# 通貨コード:GBP
# 時間帯:±0
# 夏時間:+1
# ISO 3166-1:GB / GBR
# ccTLD:.uk / .gb使用は.ukに比べ圧倒的少数。
# 国際電話番号:44
# 注記:
