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

MODEL_NAME = "gemini-2.5-flash"
SUBJECT = "global_facts"

# 全問解くなら None
MAX_QUESTIONS = 10

# 何問まとめて1回のAPIに投げるか
BATCH_SIZE = 5

# バッチごとの待ち時間
SLEEP_SECONDS = 3


# ==============================
# APIキーの読み込み
# ==============================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key is None:
    raise ValueError(
        "GEMINI_API_KEY が読み込めていません。.env ファイルを確認してください。"
    )

client = genai.Client(api_key=api_key)


def move_correct_answer_to_d(df):
    """
    正解の選択肢を必ずDに移動する関数。

    例:
    元の正解がAなら、
    Aの中身をDへ移動し、
    元のDの中身をAへ移動する。

    その結果、すべての問題の正解ラベルはDになる。
    """

    new_df = df.copy()

    for idx, row in new_df.iterrows():
        gold = str(row["answer"]).strip()

        # 元の正解選択肢の中身
        correct_text = row[gold]

        # 元のDの中身
        original_d_text = row["D"]

        # 正解がすでにDなら、そのままでよい
        if gold == "D":
            new_df.at[idx, "answer"] = "D"
            continue

        # 正解だった選択肢の場所に、元のDの中身を移す
        new_df.at[idx, gold] = original_d_text

        # Dに正解の中身を移す
        new_df.at[idx, "D"] = correct_text

        # 正解ラベルをDにする
        new_df.at[idx, "answer"] = "D"

    return new_df


# ==============================
# JMMLUデータの読み込み
# ==============================

url = f"https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/{SUBJECT}.csv"
df = pd.read_csv(url, header=None)
df.columns = ["question", "A", "B", "C", "D", "answer"]

if MAX_QUESTIONS is not None:
    df = df.head(MAX_QUESTIONS)

# 43番の実験:
# 正解の選択肢をすべてDに移動する
df = move_correct_answer_to_d(df)

# ==============================
# バッチ用プロンプトを作る関数
# ==============================


def make_batch_prompt(batch_df):
    """
    複数問をまとめてGeminiに渡すためのプロンプトを作る。
    出力形式を固定して、あとでPythonで解析しやすくする。
    """

    questions_text = ""

    for i, row in batch_df.iterrows():
        questions_text += f"""
問題ID: {i}
問題:
{row["question"]}

選択肢:
A. {row["A"]}
B. {row["B"]}
C. {row["C"]}
D. {row["D"]}
"""

    prompt = f"""
以下の4択問題に答えてください。
各問題について、正解だと思う選択肢を A, B, C, D のいずれか1文字で答えてください。

必ず次の形式だけで出力してください。
説明は不要です。

問題ID: 答え
問題ID: 答え

例:
0: A
1: C
2: D

{questions_text}
"""
    return prompt


# ==============================
# Geminiの出力を解析する関数
# ==============================


def parse_batch_output(text):
    """
    Geminiの出力から、問題IDと予測ラベルを取り出す。

    期待する出力例:
    0: A
    1: C
    2: D

    戻り値:
    {0: "A", 1: "C", 2: "D"} のような辞書
    """

    predictions = {}

    # 各行を見る
    for line in text.splitlines():
        line = line.strip()

        # 「数字: A/B/C/D」の形を探す
        match = re.match(r"^(\d+)\s*[:：]\s*([ABCD])", line)

        if match:
            question_id = int(match.group(1))
            answer = match.group(2)
            predictions[question_id] = answer

    return predictions


# ==============================
# Geminiに問い合わせる関数
# ==============================


def ask_gemini(prompt, max_retries=5):
    """
    Gemini APIに問い合わせる。
    503や429が出た場合は、少し待って再試行する。
    """

    for i in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                ),
            )
            return response.text

        except errors.ServerError as e:
            # 503 UNAVAILABLE など
            print(f"サーバーエラーが起きました。{i + 1}回目のリトライをします。")
            print(e)
            time.sleep(10 * (i + 1))

        except errors.ClientError as e:
            # 429 RESOURCE_EXHAUSTED など
            print(f"クライアントエラーが起きました。{i + 1}回目のリトライをします。")
            print(e)

            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                # 無料枠制限に当たった場合は、長めに待つ
                print("レート制限に達したため、60秒待機します。")
                time.sleep(60)
            else:
                # 429以外のエラーは別問題なので止める
                raise

    return ""


# ==============================
# バッチ単位で解答させる
# ==============================

results = []

for start in range(0, len(df), BATCH_SIZE):
    end = start + BATCH_SIZE
    batch_df = df.iloc[start:end]

    prompt = make_batch_prompt(batch_df)

    output_text = ask_gemini(prompt)
    pred_dict = parse_batch_output(output_text)

    for index, row in batch_df.iterrows():
        gold = str(row["answer"]).strip()
        pred = pred_dict.get(index)

        is_correct = pred == gold

        results.append(
            {
                "index": index,
                "question": row["question"],
                "gold": gold,
                "prediction": pred,
                "raw_output": output_text,
                "is_correct": is_correct,
            }
        )

        print(f"{index + 1}問目: 正解={gold}, 予測={pred}, 正誤={is_correct}")

    time.sleep(SLEEP_SECONDS)


# ==============================
# 正解率を表示
# ==============================

result_df = pd.DataFrame(results)

correct_count = result_df["is_correct"].sum()
total_count = len(result_df)
accuracy = correct_count / total_count

print("================================")
print(f"科目: {SUBJECT}")
print(f"問題数: {total_count}")
print(f"正解数: {correct_count}")
print(f"正解率: {accuracy:.3f}")
print("================================")


# ==============================
# CSV保存
# ==============================

output_file = f"43_{SUBJECT}_results.csv"
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"結果を {output_file} に保存しました。")

# 出力
# 1問目: 正解=D, 予測=D, 正誤=True
# 2問目: 正解=D, 予測=D, 正誤=True
# 3問目: 正解=D, 予測=D, 正誤=True
# 4問目: 正解=D, 予測=C, 正誤=False
# 5問目: 正解=D, 予測=B, 正誤=False
# 6問目: 正解=D, 予測=B, 正誤=False
# 7問目: 正解=D, 予測=D, 正誤=True
# 8問目: 正解=D, 予測=C, 正誤=False
# 9問目: 正解=D, 予測=D, 正誤=True
# 10問目: 正解=D, 予測=B, 正誤=False
# ================================
# 科目: global_facts
# 問題数: 10
# 正解数: 5
# 正解率: 0.500
# ================================
# 結果を 43_global_facts_results.csv に保存しました。

# 比較用に42番の結果も下に載せる
# 1問目: 正解=B, 予測=B, 正誤=True
# 2問目: 正解=C, 予測=C, 正誤=True
# 3問目: 正解=A, 予測=A, 正誤=True
# 4問目: 正解=C, 予測=D, 正誤=False
# 5問目: 正解=A, 予測=A, 正誤=True
# 6問目: 正解=C, 予測=B, 正誤=False
# 7問目: 正解=C, 予測=C, 正誤=True
# 8問目: 正解=C, 予測=D, 正誤=False
# 9問目: 正解=B, 予測=B, 正誤=True
# 10問目: 正解=D, 予測=B, 正誤=False
# ================================
# 科目: global_facts
# 問題数: 10
# 正解数: 6
# 正解率: 0.600
# ================================
# 結果を 42_global_facts_results.csv に保存しました。
