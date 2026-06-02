import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

# ==============================
# 設定
# ==============================

# 使用するモデル
MODEL_NAME = "gemini-2.5-flash"


# ==============================
# APIキーの読み込み
# ==============================

# .env ファイルを読み込む
load_dotenv()

# 環境変数から Gemini API キーを取得する
api_key = os.getenv("GEMINI_API_KEY")

# APIキーが読み込めなかった場合はエラーを出す
if api_key is None:
    raise ValueError(
        "GEMINI_API_KEY が読み込めていません。.env ファイルを確認してください。"
    )

# Gemini API のクライアントを作成する
client = genai.Client(api_key=api_key)


# ==============================
# プロンプトを作る
# ==============================

prompt = """
以下の問いに答えてください。
答えの駅名だけでなく、答えを選んだ理由も簡単に説明してください。

問い:
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
東急大井町線の大井町方面の電車に乗り換えたとき、
各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。
自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。
目的地の駅の名前を答えてください。
"""


# ==============================
# Geminiに問い合わせる関数
# ==============================


def ask_gemini(prompt, max_retries=5):
    """
    Gemini APIに問い合わせる関数。
    503や429などの一時的なエラーが出た場合は、少し待って再試行する。
    """

    for i in range(max_retries):
        try:
            # Gemini APIにプロンプトを送って、応答を生成させる
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    # 44番は答えが決まっている問題なので、ランダム性を低くする
                    temperature=0,
                    # 応答が長くなりすぎないように上限を設定する
                    max_output_tokens=2048,
                ),
            )

            # Geminiの応答テキストを返す
            return response.text

        except errors.ServerError as e:
            # 503 UNAVAILABLE など、Gemini側の混雑エラー
            print(f"サーバーエラーが起きました。{i + 1}回目のリトライをします。")
            print(e)

            # 10秒、20秒、30秒... と少しずつ長く待つ
            time.sleep(10 * (i + 1))

        except errors.ClientError as e:
            # 429 RESOURCE_EXHAUSTED など、レート制限エラー
            print(f"クライアントエラーが起きました。{i + 1}回目のリトライをします。")
            print(e)

            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("レート制限に達したため、60秒待機します。")
                time.sleep(60)
            else:
                # 429以外のクライアントエラーは、APIキーなど別の原因の可能性があるので止める
                raise

    # 指定回数リトライしても失敗した場合
    return "エラーにより応答を生成できませんでした。"


# ==============================
# 実行
# ==============================

answer = ask_gemini(prompt)

print("===== Geminiの応答 =====")
print(answer)

# 出力
# ===== Geminiの応答 =====
# 目的地の駅は**緑が丘駅**です。

# **理由:**
# 1.  つばめちゃんは自由が丘駅から東急大井町線の大井町方面の急行に乗車しました。
# 2.  自由が丘駅の次の東急大井町線（大井町方面）の急行停車駅は**大岡山駅**です。
# 3.  大岡山駅で降車し、反対方向（二子玉川方面）に一駅戻ると、**緑が丘駅**になります。
