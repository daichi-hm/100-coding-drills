import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-medium"

# ユーザーからの質問
QUESTION = "What do you call a sweet eaten after dinner?"

# 最大で何トークン回答を生成するか
MAX_NEW_TOKENS = 20


def apply_chat_template(messages):
    """
    messagesをチャット形式の文字列へ変換する関数。

    例：
    [
        {
            "role": "user",
            "content": "Hello."
        }
    ]

    ↓

    User: Hello.
    Assistant:
    """

    # 役割と発言を整形した文字列を順番に入れるリスト
    prompt_parts = []

    # messagesからメッセージを1つずつ取り出す
    for message in messages:
        # 誰の発言なのかを取得する
        role = message["role"]

        # 発言内容を取得する
        content = message["content"]

        # roleに応じて、文字列の先頭に名前を付ける
        if role == "system":
            prompt_parts.append(
                f"System: {content}\n"
            )

        elif role == "user":
            prompt_parts.append(
                f"User: {content}\n"
            )

        elif role == "assistant":
            prompt_parts.append(
                f"Assistant: {content}\n"
            )

        else:
            # 想定外のroleが入っていた場合はエラーにする
            raise ValueError(
                f"未対応のroleです: {role}"
            )

    # 最後にAssistant:を追加する
    #
    # これによりモデルへ、
    # 「次はAssistantの回答を生成してください」
    # と伝える
    prompt_parts.append("Assistant:")

    # リスト中の文字列をすべて連結して返す
    prompt = "".join(prompt_parts)

    return prompt


def main():
    # --------------------------------------------------
    # 1. GPUまたはCPUを選ぶ
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"使用するデバイス: {device}")

    # --------------------------------------------------
    # 2. トークナイザとモデルを読み込む
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    # モデルをGPUまたはCPUへ移動する
    model = model.to(device)

    # 評価・推論モードにする
    model.eval()

    # --------------------------------------------------
    # 3. 会話をmessagesとして用意する
    # --------------------------------------------------
    # messagesは、
    # roleとcontentを持つ辞書のリスト
    messages = [
        {
            "role": "user",
            "content": QUESTION,
        }
    ]

    # --------------------------------------------------
    # 4. チャットテンプレートを適用する
    # --------------------------------------------------
    prompt = apply_chat_template(messages)

    print("\n===== messages =====")
    print(messages)

    print("\n===== モデルに与えるプロンプト =====")
    print(prompt)

    # --------------------------------------------------
    # 5. プロンプトをトークン化する
    # --------------------------------------------------
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # input_idsやattention_maskを、
    # モデルと同じデバイスへ移動する
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # プロンプト部分のトークン数を取得する
    #
    # generate()の結果にはプロンプトも含まれるため、
    # 後から回答だけを切り出すために使う
    input_length = inputs["input_ids"].shape[1]

    # --------------------------------------------------
    # 6. Assistantの回答を生成する
    # --------------------------------------------------
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,

            # プロンプトの後ろに最大20トークン生成する
            max_new_tokens=MAX_NEW_TOKENS,

            # Greedy searchを使う
            do_sample=False,
            num_beams=1,

            # GPT-2には専用のpad_tokenがないため、
            # eos_tokenのIDをpaddingにも使う
            pad_token_id=tokenizer.eos_token_id,
        )

    # --------------------------------------------------
    # 7. 生成された回答部分だけを取り出す
    # --------------------------------------------------
    # generated_ids[0]には、
    #
    # [プロンプト部分 + 新しく生成された回答]
    #
    # が入っている
    generated_sequence = generated_ids[0]

    # プロンプトの後ろだけを切り出す
    response_ids = generated_sequence[input_length:]

    # 回答部分のトークンIDを文字列へ戻す
    response = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n===== 生成された応答 =====")
    print(response)


if __name__ == "__main__":
    main()

#出力
# 使用するデバイス: cuda

# ===== messages =====
# [{'role': 'user', 'content': 'What do you call a sweet eaten after dinner?'}]

# ===== モデルに与えるプロンプト =====
# User: What do you call a sweet eaten after dinner?
# Assistant:

# ===== 生成された応答 =====
#  A sweet eaten after dinner.
# Assistant: A sweet eaten after dinner.
# Assistant: A sweet