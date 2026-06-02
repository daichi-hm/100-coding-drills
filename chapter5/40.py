import os
from dotenv import load_dotenv
from google import genai

# .envを読み込む
load_dotenv()

# 環境変数からAPIキーを取得
api_key = os.getenv("GEMINI_API_KEY")

if api_key is None:
    raise ValueError(
        "GEMINI_API_KEY が読み込めていません。.env ファイルを確認してください。"
    )


client = genai.Client(api_key=api_key)

prompt = """
以下の日本史の問題に答えてください。
年代の古い順に、ア・イ・ウの記号だけを「イ→ウ→ア」のような形式で出力してください。

9世紀に活躍した人物に関係するできごとについて述べた次のア~ウを年代の古い順に正しく並べよ。

ア 藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ 嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ 藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

# 結果の表示
print("解答:")
print(response.text)

# 出力
# 解答:
# イ→ウ→ア
