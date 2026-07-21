import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-medium"

# この文章に続くテキストを生成する
PROMPT = "The movie was full of"

# プロンプトの後ろに最大何トークン生成するか
# 長すぎると確率の表示が多くなるため、10トークンにする
MAX_NEW_TOKENS = 10


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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # モデルをGPUまたはCPUへ移動する
    model = model.to(device)

    # 推論モードにする
    model.eval()

    # --------------------------------------------------
    # 3. プロンプトをトークン化する
    # --------------------------------------------------
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    )

    # 入力データもモデルと同じデバイスへ移動する
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # プロンプトのトークン数を取得する
    #
    # input_idsのshapeは、
    # [バッチサイズ, トークン数]
    #
    # shape[1]なので、トークン数を取り出している
    input_length = inputs["input_ids"].shape[1]

    print(f"プロンプト: {PROMPT!r}")
    print(f"プロンプトのトークン数: {input_length}")

    # --------------------------------------------------
    # 4. 続きのテキストを生成する
    # --------------------------------------------------
    with torch.no_grad():
        generation_output = model.generate(
            # input_idsやattention_maskを渡す
            **inputs,

            # 最大10トークン生成する
            max_new_tokens=MAX_NEW_TOKENS,

            # Greedy searchを使う
            # 各時点で最も確率が高いトークンを選ぶ
            do_sample=False,
            num_beams=1,

            # GPT-2にはpad_tokenが設定されていないため、
            # eos_tokenのIDをpaddingにも利用する
            pad_token_id=tokenizer.eos_token_id,

            # 通常のTensorだけではなく、
            # sequencesやscoresを含む形式で結果を返す
            return_dict_in_generate=True,

            # 各生成ステップの予測スコアも返す
            output_scores=True,
        )

    # --------------------------------------------------
    # 5. 生成された全文を取り出す
    # --------------------------------------------------
    # sequencesには、
    # [プロンプトのトークン + 新しく生成されたトークン]
    # が入っている
    generated_sequence = generation_output.sequences[0]

    # トークンID列を人間が読める文章に戻す
    generated_text = tokenizer.decode(
        generated_sequence,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n===== 生成されたテキスト =====")
    print(generated_text)

    # --------------------------------------------------
    # 6. 新しく生成されたトークンだけを取り出す
    # --------------------------------------------------
    # プロンプト部分は最初のinput_length個なので、
    # それより後ろだけを取り出す
    generated_token_ids = generated_sequence[input_length:]

    # --------------------------------------------------
    # 7. 各トークンの確率を表示する
    # --------------------------------------------------
    print("\n===== 生成された各トークンの尤度 =====")

    # 生成されたテキスト全体の対数尤度を計算するための変数
    total_log_likelihood = 0.0

    # generation_output.scoresには、
    # 各生成ステップのスコアが順番に入っている
    #
    # scores[0]：1個目の生成トークンを選ぶときのスコア
    # scores[1]：2個目の生成トークンを選ぶときのスコア
    # ...
    for step, token_scores in enumerate(generation_output.scores):
        # このステップで実際に生成されたトークンID
        token_id = generated_token_ids[step].item()

        # token_scoresのshapeは、
        # [バッチサイズ, 語彙数]
        #
        # 今回は1文章だけなので、[0]で先頭の文章を取り出す
        scores_for_one_sequence = token_scores[0]

        # 全トークンのスコアを確率へ変換する
        token_probabilities = torch.softmax(
            scores_for_one_sequence,
            dim=-1,
        )

        # 実際に選ばれたトークンの確率を取り出す
        probability = token_probabilities[token_id].item()

        # 確率の対数を累積する
        total_log_likelihood += math.log(probability)

        # GPT-2内部で使われるトークン表記
        raw_token = tokenizer.convert_ids_to_tokens(token_id)

        # 人間が読みやすい文字列へ戻す
        decoded_token = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )

        print(
            f"{step + 1:2d}番目: "
            f"token={decoded_token!r}, "
            f"raw_token={raw_token!r}, "
            f"token_id={token_id:5d}, "
            f"尤度={probability:.6f} "
            f"({probability * 100:.3f}%)"
        )

    # --------------------------------------------------
    # 8. 生成テキスト全体の尤度も表示する
    # --------------------------------------------------
    # 各トークンの条件付き確率を掛け合わせたものが、
    # 生成された続き全体の尤度になる
    sequence_likelihood = math.exp(total_log_likelihood)

    print("\n===== 続き全体の尤度 =====")
    print(f"対数尤度: {total_log_likelihood:.6f}")
    print(f"尤度    : {sequence_likelihood:.12e}")


if __name__ == "__main__":
    main()

#出力
# 使用するデバイス: cuda
# プロンプト: 'The movie was full of'
# プロンプトのトークン数: 5

# ===== 生成されたテキスト =====
# The movie was full of great moments, but the most memorable was when the

# ===== 生成された各トークンの尤度 =====
#  1番目: token=' great', raw_token='Ġgreat', token_id= 1049, 尤度=0.023094 (2.309%)
#  2番目: token=' moments', raw_token='Ġmoments', token_id= 7188, 尤度=0.188108 (18.811%)
#  3番目: token=',', raw_token=',', token_id=   11, 尤度=0.363473 (36.347%)
#  4番目: token=' but', raw_token='Ġbut', token_id=  475, 尤度=0.222480 (22.248%)
#  5番目: token=' the', raw_token='Ġthe', token_id=  262, 尤度=0.136633 (13.663%)
#  6番目: token=' most', raw_token='Ġmost', token_id=  749, 尤度=0.080318 (8.032%)
#  7番目: token=' memorable', raw_token='Ġmemorable', token_id=18078, 尤度=0.304252 (30.425%)
#  8番目: token=' was', raw_token='Ġwas', token_id=  373, 尤度=0.179842 (17.984%)
#  9番目: token=' when', raw_token='Ġwhen', token_id=  618, 尤度=0.259740 (25.974%)
# 10番目: token=' the', raw_token='Ġthe', token_id=  262, 尤度=0.106244 (10.624%)

# ===== 続き全体の尤度 =====
# 対数尤度: -18.961769
# 尤度    : 5.821144102287e-09