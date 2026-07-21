import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_NAME = "openai-community/gpt2-medium"

# この文章に続くテキストを生成する
PROMPT = "The movie was full of"

# プロンプトの後ろに最大何トークン生成するか
MAX_NEW_TOKENS = 30


def generate_and_print(
    title,
    model,
    tokenizer,
    inputs,
    **generation_settings,
):
    """
    指定された生成方法で文章を生成し、
    生成結果を表示する関数

    generation_settingsには、
    do_sample、temperature、num_beamsなどを渡す
    """

    print(f"\n{'=' * 15} {title} {'=' * 15}")

    # 今回は学習しないため、勾配を計算しない
    with torch.no_grad():
        generated_ids = model.generate(
            # プロンプトのinput_idsやattention_maskを渡す
            **inputs,

            # プロンプトに続けて、最大30トークン生成する
            max_new_tokens=MAX_NEW_TOKENS,

            # GPT-2には通常pad_tokenがないため、
            # 文末トークンをpad_tokenとして使用する
            pad_token_id=tokenizer.eos_token_id,

            # 生成方法ごとの設定を渡す
            **generation_settings,
        )

    # プロンプトが何トークンだったかを取得する
    input_length = inputs["input_ids"].shape[1]

    # generated_idsには、
    # [プロンプトのトークン + 新しく生成されたトークン]
    # が入っている
    for index, sequence_ids in enumerate(generated_ids):
        # プロンプトを含む文章全体を文字列へ戻す
        full_text = tokenizer.decode(
            sequence_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # プロンプト部分を除き、
        # 新しく生成されたトークンだけを取り出す
        continuation_ids = sequence_ids[input_length:]

        # 生成された続きだけを文字列へ戻す
        continuation_text = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        print(f"\n候補{index + 1}")
        print(f"続きだけ: {continuation_text!r}")
        print(f"全文      : {full_text}")


def main():
    # GPUが使える場合はGPU、使えない場合はCPUを使う
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"使用するデバイス: {device}")

    # トークナイザを読み込む
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 mediumを読み込む
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # モデルをGPUまたはCPUへ移動する
    model = model.to(device)

    # 評価・推論モードにする
    model.eval()

    # プロンプトをトークン化する
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    )

    # input_idsやattention_maskを、
    # モデルと同じGPUまたはCPUへ移動する
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print(f"プロンプト: {PROMPT!r}")
    print(f"プロンプトのトークン数: {inputs['input_ids'].shape[1]}")

    # --------------------------------------------------
    # 1. Greedy search
    # --------------------------------------------------
    # 各時点で、最も確率が高いトークンを選ぶ
    #
    # ランダム性がないため、同じ入力なら基本的に
    # 毎回同じ文章が生成される
    generate_and_print(
        title="Greedy search",
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        do_sample=False,
        num_beams=1,
        num_return_sequences=1,
    )

    # --------------------------------------------------
    # 2. Beam search
    # --------------------------------------------------
    # 5個の文章候補を並行して保持しながら生成し、
    # その中からスコアの高い3候補を返す
    generate_and_print(
        title="Beam search",
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        do_sample=False,
        num_beams=5,
        num_return_sequences=3,
        early_stopping=True,
    )

    # --------------------------------------------------
    # 3. Sampling
    # --------------------------------------------------
    # 確率分布に従ってランダムにトークンを選ぶ
    #
    # temperatureを変え、
    # 生成結果の変化を確認する
    temperatures = [0.5, 1.0, 1.5]

    for temperature in temperatures:
        # Samplingにはランダム性があるため、
        # 比較しやすいように乱数の種を固定する
        set_seed(42)

        generate_and_print(
            title=f"Sampling: temperature={temperature}",
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,

            # 確率的にトークンを選ぶ
            do_sample=True,

            # 確率分布の鋭さを調整する
            temperature=temperature,

            # 確率が高い上位50トークンを
            # Samplingの候補とする
            top_k=50,

            # それぞれのtemperatureで3文章生成する
            num_return_sequences=3,
        )


if __name__ == "__main__":
    main()

#出力
# 使用するデバイス: cuda
# プロンプト: 'The movie was full of'
# プロンプトのトークン数: 5

# =============== Greedy search ===============

# 候補1
# 続きだけ: ' great moments, but the most memorable was when the characters were reunited.\n\nThe movie was full of great moments, but the most memorable was when'
# 全文      : The movie was full of great moments, but the most memorable was when the characters were reunited.

# The movie was full of great moments, but the most memorable was when

# =============== Beam search ===============

# 候補1
# 続きだけ: ' great moments, but it was also full of bad ones.\n\nThe movie was full of great moments, but it was also full of bad ones'
# 全文      : The movie was full of great moments, but it was also full of bad ones.

# The movie was full of great moments, but it was also full of bad ones

# 候補2
# 続きだけ: ' great moments, but it also had a lot of bad ones.\n\nThe movie was full of great moments, but it also had a lot of'
# 全文      : The movie was full of great moments, but it also had a lot of bad ones.

# The movie was full of great moments, but it also had a lot of

# 候補3
# 続きだけ: ' great moments, but it was also full of bad ones.\n\nThe movie was full of great moments.\n\nThe movie was full of great'
# 全文      : The movie was full of great moments, but it was also full of bad ones.

# The movie was full of great moments.

# The movie was full of great

# =============== Sampling: temperature=0.5 ===============

# 候補1
# 続きだけ: ' action, but the action was muted, and it was hard to see why it was not a hit.\n\nThe most important thing, though,'
# 全文      : The movie was full of action, but the action was muted, and it was hard to see why it was not a hit.

# The most important thing, though,

# 候補2
# 続きだけ: ' the usual, but it was the first time that I had seen the movie with both eyes open. I remember thinking to myself, "This is going'
# 全文      : The movie was full of the usual, but it was the first time that I had seen the movie with both eyes open. I remember thinking to myself, "This is going

# 候補3
# 続きだけ: " character development, and it was a fun ride. But I'm not going to give it to you to be a definitive review.\n\nThe movie"
# 全文      : The movie was full of character development, and it was a fun ride. But I'm not going to give it to you to be a definitive review.

# The movie

# =============== Sampling: temperature=1.0 ===============

# 候補1
# 続きだけ: ' action, explosions and moments that felt surreal and realistic. It helped me relax when I felt anxious or guilty. The only thing I could say about this'
# 全文      : The movie was full of action, explosions and moments that felt surreal and realistic. It helped me relax when I felt anxious or guilty. The only thing I could say about this

# 候補2
# 続きだけ: ' memorable scenes. The crew of the Starship Ego had seen the film and it had been set to music when the asteroid, a part of the asteroid'
# 全文      : The movie was full of memorable scenes. The crew of the Starship Ego had seen the film and it had been set to music when the asteroid, a part of the asteroid

# 候補3
# 続きだけ: " character development, great characters, and characters who could be made to be relatable to people who don't get to see them often. I think there"
# 全文      : The movie was full of character development, great characters, and characters who could be made to be relatable to people who don't get to see them often. I think there

# =============== Sampling: temperature=1.5 ===============

# 候補1
# 続きだけ: ' these "tru" situations — when Mr. White would show an unusual gesture and not immediately let off the leash — all meant to convey the impression'
# 全文      : The movie was full of these "tru" situations — when Mr. White would show an unusual gesture and not immediately let off the leash — all meant to convey the impression

# 候補2
# 続きだけ: ' memorable scenes. We were all impressed with how much of a lot, so it certainly gave us more of an incentive to watch for every one of our'
# 全文      : The movie was full of memorable scenes. We were all impressed with how much of a lot, so it certainly gave us more of an incentive to watch for every one of our

# 候補3
# 続きだけ: ' character performances, great characters, and characters who deserved more to be included within those characters. It had a ton of great actors and really did the audience'
# 全文      : The movie was full of character performances, great characters, and characters who deserved more to be included within those characters. It had a ton of great actors and really did the audience