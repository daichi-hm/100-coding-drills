import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-medium"

# パープレキシティを測定する4文
#
# 上の2文は文法的に自然
# 下の2文は主語とbe動詞の単数・複数が一致していない
SENTENCES = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises",
]


def calculate_perplexity(
    text,
    model,
    tokenizer,
    device,
):
    """
    1つの文章を受け取り、
    GPT-2 mediumによるパープレキシティを計算する関数
    """

    # --------------------------------------------------
    # 1. 文章をトークン化する
    # --------------------------------------------------
    # 文章をGPT-2が扱えるトークンIDのTensorへ変換する
    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    # input_idsやattention_maskを、
    # モデルと同じGPUまたはCPUへ移動する
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------
    # 2. 正解ラベルを用意する
    # --------------------------------------------------
    # GPT-2には、
    #
    # 入力された文章を見ながら、
    # その文章中の次トークンを予測してもらう
    #
    # そのため、入力されたトークンID列を
    # そのまま正解ラベルとして使用する
    labels = inputs["input_ids"].clone()

    # --------------------------------------------------
    # 3. 平均負の対数尤度を計算する
    # --------------------------------------------------
    # 今回は学習しないので、勾配を計算しない
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )

    # labelsを渡すと、GPT-2が内部で
    # 次トークン予測の平均損失を計算してくれる
    loss = outputs.loss

    # --------------------------------------------------
    # 4. 損失からパープレキシティを計算する
    # --------------------------------------------------
    # lossは平均負の対数尤度なので、
    # expを適用するとパープレキシティになる
    perplexity = torch.exp(loss)

    # 表示用に、トークンIDをPythonのリストへ変換する
    token_ids = inputs["input_ids"][0].tolist()

    # 各トークンIDを人間が読める文字列へ戻す
    decoded_tokens = [
        tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]

    # Tensorから通常のPythonの数値を取り出して返す
    return (
        loss.item(),
        perplexity.item(),
        token_ids,
        decoded_tokens,
    )


def main():
    # --------------------------------------------------
    # 1. GPUまたはCPUを選ぶ
    # --------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"使用するデバイス: {device}")

    # --------------------------------------------------
    # 2. トークナイザを読み込む
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --------------------------------------------------
    # 3. GPT-2 mediumを読み込む
    # --------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # モデルをGPUまたはCPUへ移動する
    model = model.to(device)

    # 評価・推論モードにする
    model.eval()

    print("\n===== パープレキシティの測定結果 =====")

    # --------------------------------------------------
    # 4. 4文を1文ずつ評価する
    # --------------------------------------------------
    for index, sentence in enumerate(SENTENCES):
        (
            loss,
            perplexity,
            token_ids,
            decoded_tokens,
        ) = calculate_perplexity(
            text=sentence,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        print(f"\n文{index + 1}")
        print(f"文章              : {sentence}")
        print(f"トークン          : {decoded_tokens}")
        print(f"トークンID        : {token_ids}")
        print(f"平均負の対数尤度  : {loss:.6f}")
        print(f"パープレキシティ  : {perplexity:.6f}")


if __name__ == "__main__":
    main()

#出力
# 使用するデバイス: cuda

# ===== パープレキシティの測定結果 =====

# 文1
# 文章              : The movie was full of surprises
# トークン          : ['The', ' movie', ' was', ' full', ' of', ' surprises']
# トークンID        : [464, 3807, 373, 1336, 286, 24072]
# 平均負の対数尤度  : 4.493728
# パープレキシティ  : 89.454277

# 文2
# 文章              : The movies were full of surprises
# トークン          : ['The', ' movies', ' were', ' full', ' of', ' surprises']
# トークンID        : [464, 6918, 547, 1336, 286, 24072]
# 平均負の対数尤度  : 5.105276
# パープレキシティ  : 164.889511

# 文3
# 文章              : The movie were full of surprises
# トークン          : ['The', ' movie', ' were', ' full', ' of', ' surprises']
# トークンID        : [464, 3807, 547, 1336, 286, 24072]
# 平均負の対数尤度  : 5.781083
# パープレキシティ  : 324.109894

# 文4
# 文章              : The movies was full of surprises
# トークン          : ['The', ' movies', ' was', ' full', ' of', ' surprises']
# トークンID        : [464, 6918, 373, 1336, 286, 24072]
# 平均負の対数尤度  : 5.962157
# パープレキシティ  : 388.447205