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

# 川柳のお題
THEME = "研究室生活"


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

prompt = f"""
お題「{THEME}」について、川柳の案を10個作成してください。

条件:
- 川柳は「五・七・五」のリズムを意識してください。
- 10個すべて日本語で作成してください。
- それぞれ番号を付けてください。
- 説明は不要です。
- 出力は川柳だけにしてください。
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
                    # 川柳生成は創作タスクなので、少しだけランダム性を持たせる
                    temperature=0.8,
                    # 10個の川柳を出力するので、少し余裕を持たせる
                    max_output_tokens=2048,
                ),
            )

            # Geminiの応答テキストを返す
            return response.text

        except errors.ServerError as e:
            # 503 UNAVAILABLE など、Gemini側の混雑エラー
            print(f"サーバーエラーが起きました。{i + 1}回目のリトライをします。")
            print(e)

            # 10秒、20秒、30秒... と待ち時間を少しずつ長くする
            time.sleep(10 * (i + 1))

        except errors.ClientError as e:
            # 429 RESOURCE_EXHAUSTED など、レート制限エラー
            print(f"クライアントエラーが起きました。{i + 1}回目のリトライをします。")
            print(e)

            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("レート制限に達したため、60秒待機します。")
                time.sleep(60)
            else:
                # 429以外のクライアントエラーは、APIキーなど別原因の可能性があるので止める
                raise

    # 指定回数リトライしても失敗した場合
    return "エラーにより川柳を生成できませんでした。"


# ==============================
# 実行
# ==============================

senryu = ask_gemini(prompt)

print("===== 川柳生成結果 =====")
print(f"お題: {THEME}")
print(senryu)

# 出力
# ===== 川柳生成結果 =====
# お題: 研究室生活
# 1. 徹夜明け コーヒー片手に 夢うつつ
# 2. データ出ず 頭抱える 夜が更ける
# 3. 教授室 ドア開く音に 背筋伸び
# 4. 論文を 眺め五月雨 過ぎていく
# 5. 実験は 失敗重ねて 進む道
# 6. パソコンと 語らい迎える 朝日かな
# 7. 締め切り前 仲間と食べる カップ麺
# 8. 白衣着て 今日もフラスコ 振る一日
# 9. 進捗は 聞かれるたびに 汗をかく
# 10. 卒業の 夢見て今日も キーを打つ
