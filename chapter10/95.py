import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-medium"

# 94番で使用した最初の質問
FIRST_QUESTION = (
    "What do you call a sweet eaten after dinner?"
)

# 95番で追加する質問
FOLLOW_UP_QUESTION = (
    "Please give me the plural form of the word "
    "with its spelling in reverse order."
)

# 1回の応答で最大何トークン生成するか
MAX_NEW_TOKENS = 20


def build_chat_prompt(messages):
    """
    messagesに入っている会話履歴を、
    GPT-2に渡す1本の文字列へ変換する関数。
    """

    # 整形した各メッセージを順番に入れるリスト
    prompt_parts = []

    # 会話履歴からメッセージを1つずつ取り出す
    for message in messages:
        # 発言者の役割を取得
        role = message["role"]

        # 発言内容を取得
        content = message["content"]

        # ユーザーの発言
        if role == "user":
            prompt_parts.append(
                f"User: {content}\n"
            )

        # アシスタントの発言
        elif role == "assistant":
            prompt_parts.append(
                f"Assistant: {content}\n"
            )

        # systemの発言を使う場合
        elif role == "system":
            prompt_parts.append(
                f"System: {content}\n"
            )

        # 想定外のroleが指定された場合
        else:
            raise ValueError(
                f"未対応のroleです: {role}"
            )

    # 最後にAssistant:を追加する
    prompt_parts.append("Assistant:")

    # リスト中の文字列を1本につなげる
    prompt = "".join(prompt_parts)

    return prompt


def generate_response(
    prompt,
    model,
    tokenizer,
    device,
):
    """
    1つのプロンプトを受け取り、
    GPT-2でその続きを生成する関数。
    """

    # --------------------------------------------------
    # 1. プロンプトをトークン化する
    # --------------------------------------------------
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # input_idsとattention_maskを
    # モデルと同じGPUまたはCPUへ移動する
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # プロンプト部分のトークン数を取得する
    #
    # generate()の結果には、
    # プロンプトと生成部分の両方が入るため、
    # 後で生成部分だけを切り出すために使う
    input_length = inputs["input_ids"].shape[1]

    # --------------------------------------------------
    # 2. 応答を生成する
    # --------------------------------------------------
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,

            # 最大20トークン生成する
            max_new_tokens=MAX_NEW_TOKENS,

            # Greedy search
            do_sample=False,
            num_beams=1,

            # GPT-2には専用のpad_tokenがないため、
            # eos_tokenをpaddingにも使う
            pad_token_id=tokenizer.eos_token_id,
        )

    # --------------------------------------------------
    # 3. 新しく生成された部分だけを取り出す
    # --------------------------------------------------
    # generated_idsのshapeは、
    # [生成系列数, 全トークン数]
    #
    # 今回は1系列だけなので[0]を指定する
    generated_sequence = generated_ids[0]

    # プロンプト部分より後ろだけを取り出す
    response_ids = generated_sequence[input_length:]

    # トークンIDを人間が読める文字列へ戻す
    response = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return response


def extract_one_turn(response):
    """
    GPT-2が次のAssistant:などまで生成した場合に、
    最初の応答部分だけを取り出す関数。

    例：
    Dessert.
    Assistant: Dessert.

    ↓

    Dessert.
    """

    # 次の発言者を示す文字列が生成された場合、
    # その直前までを今回の応答として扱う
    stop_markers = [
        "\nUser:",
        "\nAssistant:",
        "\nSystem:",
    ]

    for marker in stop_markers:
        if marker in response:
            response = response.split(
                marker,
                maxsplit=1,
            )[0]

    # 先頭・末尾の余分な空白や改行を削除する
    response = response.strip()

    return response


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

    # 推論・評価モードにする
    model.eval()

    # --------------------------------------------------
    # 3. 最初のユーザーメッセージを用意する
    # --------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": FIRST_QUESTION,
        }
    ]

    # --------------------------------------------------
    # 4. 1ターン目のプロンプトを作る
    # --------------------------------------------------
    first_prompt = build_chat_prompt(messages)

    print("\n===== 1ターン目のプロンプト =====")
    print(first_prompt)

    # --------------------------------------------------
    # 5. 1ターン目の応答を生成する
    # --------------------------------------------------
    first_response = generate_response(
        prompt=first_prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # GPT-2が次の会話まで生成した場合、
    # 最初の応答だけを取り出す
    first_response = extract_one_turn(
        first_response
    )

    print("\n===== 1ターン目の応答 =====")
    print(first_response)

    # --------------------------------------------------
    # 6. 生成された応答を会話履歴に追加する
    # --------------------------------------------------
    messages.append(
        {
            "role": "assistant",
            "content": first_response,
        }
    )

    # --------------------------------------------------
    # 7. 追加のユーザーメッセージを履歴へ追加する
    # --------------------------------------------------
    messages.append(
        {
            "role": "user",
            "content": FOLLOW_UP_QUESTION,
        }
    )

    # この時点でmessagesの中には、
    #
    # 1. 最初のユーザー質問
    # 2. GPT-2が生成した最初の回答
    # 3. ユーザーからの追加質問
    #
    # の3メッセージが入っている

    print("\n===== 2ターン目のmessages =====")
    print(messages)

    # --------------------------------------------------
    # 8. 会話履歴全体から2ターン目のプロンプトを作る
    # --------------------------------------------------
    second_prompt = build_chat_prompt(messages)

    print("\n===== 2ターン目のプロンプト =====")
    print(second_prompt)

    # --------------------------------------------------
    # 9. 追加質問への応答を生成する
    # --------------------------------------------------
    second_response = generate_response(
        prompt=second_prompt,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # 最初の応答部分だけを取り出す
    second_response = extract_one_turn(
        second_response
    )

    print("\n===== 2ターン目の応答 =====")
    print(second_response)


if __name__ == "__main__":
    main()

#出力
# 使用するデバイス: cuda

# ===== 1ターン目のプロンプト =====
# User: What do you call a sweet eaten after dinner?
# Assistant:

# ===== 1ターン目の応答 =====
# A sweet eaten after dinner.

# ===== 2ターン目のmessages =====
# [{'role': 'user', 'content': 'What do you call a sweet eaten after dinner?'}, {'role': 'assistant', 'content': 'A sweet eaten after dinner.'}, {'role': 'user', 'content': 'Please give me the plural form of the word with its spelling in reverse order.'}]

# ===== 2ターン目のプロンプト =====
# User: What do you call a sweet eaten after dinner?
# Assistant: A sweet eaten after dinner.
# User: Please give me the plural form of the word with its spelling in reverse order.
# Assistant:

# ===== 2ターン目の応答 =====
# Sweet eaten after dinner.