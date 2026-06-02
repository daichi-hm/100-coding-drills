import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

# ==============================
# 設定
# ==============================

# 使用するモデル
MODEL_NAME = "gemini-2.5-flash"

# 評価を何回繰り返すか
NUM_TRIALS = 5

# APIを連続で叩きすぎないように待つ秒数
SLEEP_SECONDS = 5


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
# 46番で生成した川柳
# ==============================

# ここには、46番で実際に生成された川柳を入れる
senryu_list = [
    "徹夜明け コーヒー片手に 夢うつつ",
    "データ出ず 頭抱える 夜が更ける",
    "教授室 ドア開く音に 背筋伸び",
    "論文を 眺め五月雨 過ぎていく",
    "実験は 失敗重ねて 進む道",
    "パソコンと 語らい迎える 朝日かな",
    "締め切り前 仲間と食べる カップ麺",
    "白衣着て 今日もフラスコ 振る一日",
    "進捗は 聞かれるたびに 汗をかく",
    "卒業の 夢見て今日も キーを打つ",
]


# ==============================
# 評価用プロンプトを作る関数
# ==============================


def make_evaluation_prompt(senryu_list):
    """
    川柳リストを受け取り、LLMに10段階評価させるためのプロンプトを作る関数。
    """

    # 川柳を番号付きの文字列にする
    senryu_text = ""

    for i, senryu in enumerate(senryu_list, start=1):
        senryu_text += f"{i}. {senryu}\n"

    # LLMに評価基準と出力形式を指定する
    # 48番では平均や分散を計算したいので、理由は出力させず点数だけにする
    prompt = f"""
以下の川柳の面白さを10段階で評価してください。

評価基準:
- 1点: ほとんど面白くない
- 5点: 普通
- 10点: とても面白い
- 共感しやすさ、意外性、言葉のうまさ、オチの良さを総合的に見て評価してください。

出力形式:
必ず次の形式だけで出力してください。
理由や説明は書かないでください。

1: 点数
2: 点数
3: 点数

例:
1: 8
2: 6
3: 9

川柳:
{senryu_text}
"""
    return prompt


# ==============================
# Geminiの出力から点数を取り出す関数
# ==============================


def parse_scores(text):
    """
    Geminiの出力から、川柳番号と点数を取り出す関数。

    期待する出力例:
    1: 8
    2: 6
    3: 9

    戻り値:
    {1: 8, 2: 6, 3: 9} のような辞書
    """

    scores = {}

    for line in text.splitlines():
        line = line.strip()

        # 「番号: 点数」の形を探す
        # 点数は 1〜10 を想定する
        match = re.match(r"^(\d+)\s*[:：]\s*(10|[1-9])\b", line)

        if match:
            senryu_id = int(match.group(1))
            score = int(match.group(2))
            scores[senryu_id] = score

    return scores


# ==============================
# Geminiに問い合わせる関数
# ==============================


def ask_gemini(prompt, max_retries=5):
    """
    Gemini APIに問い合わせる関数。
    503や429が出た場合は、少し待って再試行する。
    """

    for i in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    # 今回は同じ評価を複数回行い、評価の揺れを見る
                    # temperature=0 にすると比較的安定しやすい
                    temperature=0,
                    max_output_tokens=4096,
                ),
            )

            return response.text

        except errors.ServerError as e:
            print(f"サーバーエラーが起きました。{i + 1}回目のリトライをします。")
            print(e)
            time.sleep(10 * (i + 1))

        except errors.ClientError as e:
            print(f"クライアントエラーが起きました。{i + 1}回目のリトライをします。")
            print(e)

            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("レート制限に達したため、60秒待機します。")
                time.sleep(60)
            else:
                raise

    return ""


# ==============================
# 同じ評価を複数回実行する
# ==============================

results = []

for trial in range(1, NUM_TRIALS + 1):
    print(f"===== {trial}回目の評価 =====")

    # 評価用プロンプトを作る
    prompt = make_evaluation_prompt(senryu_list)

    # Geminiに評価させる
    output_text = ask_gemini(prompt)

    # Geminiの生の出力を表示
    print(output_text)

    # 出力から点数を取り出す
    scores = parse_scores(output_text)

    # 10個の川柳について、点数を保存する
    for senryu_id, senryu in enumerate(senryu_list, start=1):
        score = scores.get(senryu_id)

        results.append(
            {
                "trial": trial,
                "senryu_id": senryu_id,
                "senryu": senryu,
                "score": score,
                "raw_output": output_text,
            }
        )

    # レート制限対策として少し待つ
    time.sleep(SLEEP_SECONDS)


# ==============================
# 結果をDataFrameにする
# ==============================

result_df = pd.DataFrame(results)

# 点数がうまく取れなかった件数を確認
none_count = result_df["score"].isna().sum()

print("================================")
print(f"点数を取得できなかった件数: {none_count}")
print("================================")


# ==============================
# 川柳ごとの平均・分散・標準偏差を計算
# ==============================

summary_df = (
    result_df.groupby(["senryu_id", "senryu"])["score"]
    .agg(["mean", "var", "std"])
    .reset_index()
)

print("===== 川柳ごとの平均・分散・標準偏差 =====")
print(summary_df)


# ==============================
# CSV保存
# ==============================

result_file = "48_senryu_evaluation_results_normal.csv"
summary_file = "48_senryu_evaluation_summary_normal.csv"

result_df.to_csv(result_file, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")

print(f"詳細結果を {result_file} に保存しました。")
print(f"川柳ごとの集計結果を {summary_file} に保存しました。")

# 出力
# ===== 1回目の評価 =====
# 1: 7
# 2: 7
# 3: 8
# 4: 7
# 5: 8
# 6: 8
# 7: 7
# 8: 5
# 9: 8
# 10: 7
# ===== 2回目の評価 =====
# 1: 6
# 2: 5
# 3: 6
# 4: 7
# 5: 6
# 6: 7
# 7: 8
# 8: 5
# 9: 6
# 10: 6
# ===== 3回目の評価 =====
# 1: 7
# 2: 7
# 3: 8
# 4: 7
# 5: 8
# 6: 8
# 7: 7
# 8: 5
# 9: 8
# 10: 7
# ===== 4回目の評価 =====
# 1: 6
# 2: 5
# 3: 6
# 4: 7
# 5: 6
# 6: 7
# 7: 8
# 8: 5
# 9: 6
# 10: 6
# ===== 5回目の評価 =====
# 1: 6
# 2: 5
# 3: 6
# 4: 7
# 5: 6
# 6: 7
# 7: 8
# 8: 5
# 9: 6
# 10: 6
# ================================
# 点数を取得できなかった件数: 0
# ================================
# ===== 川柳ごとの平均・分散・標準偏差 =====
#    senryu_id             senryu  mean  var       std
# 0          1  徹夜明け コーヒー片手に 夢うつつ   6.4  0.3  0.547723
# 1          2   データ出ず 頭抱える 夜が更ける   5.8  1.2  1.095445
# 2          3    教授室 ドア開く音に 背筋伸び   6.8  1.2  1.095445
# 3          4    論文を 眺め五月雨 過ぎていく   7.0  0.0  0.000000
# 4          5      実験は 失敗重ねて 進む道   6.8  1.2  1.095445
# 5          6  パソコンと 語らい迎える 朝日かな   7.4  0.3  0.547723
# 6          7  締め切り前 仲間と食べる カップ麺   7.6  0.3  0.547723
# 7          8  白衣着て 今日もフラスコ 振る一日   5.0  0.0  0.000000
# 8          9   進捗は 聞かれるたびに 汗をかく   6.8  1.2  1.095445
# 9         10   卒業の 夢見て今日も キーを打つ   6.4  0.3  0.547723
# 詳細結果を 48_senryu_evaluation_results_normal.csv に保存しました。
# 川柳ごとの集計結果を 48_senryu_evaluation_summary_normal.csv に保存しました。
