import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 使用する事前学習済みモデル
MODEL_NAME = "openai-community/gpt2-medium"

# GPT-2に入力するプロンプト
PROMPT = "The movie was full of"

# 表示する次トークン候補の数
TOP_K = 10


def main():
    # GPUが利用できる場合はGPUを使い、
    # 利用できない場合はCPUを使う
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"使用するデバイス: {device}")

    # --------------------------------------------------
    # 1. トークナイザを読み込む
    # --------------------------------------------------
    # トークナイザは、文章をトークンIDの列に変換する
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --------------------------------------------------
    # 2. GPT-2 mediumを読み込む
    # --------------------------------------------------
    # AutoModelForCausalLMは、GPT-2のような
    # 「左側の文脈から次トークンを予測するモデル」を読み込むクラス
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # モデルをGPUまたはCPUへ移動する
    model = model.to(device)

    # 推論モードに切り替える
    # Dropoutなど、学習時だけ使用する処理が無効になる
    model.eval()

    # --------------------------------------------------
    # 3. プロンプトをトークン化する
    # --------------------------------------------------
    # return_tensors="pt"を指定すると、
    # 結果がPyTorchのTensorとして返される
    inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
    )

    # input_idsの先頭のバッチを取り出し、
    # Pythonのリストへ変換する
    input_ids = inputs["input_ids"][0].tolist()

    # トークンIDを、GPT-2内部で使われるトークン表記へ変換する
    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 各トークンIDを、人間が読みやすい文字列へ戻す
    decoded_tokens = [
        tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )
        for token_id in input_ids
    ]

    # --------------------------------------------------
    # 4. トークン化の結果を表示する
    # --------------------------------------------------
    print("\n===== プロンプトのトークン化結果 =====")
    print(f"プロンプト: {PROMPT!r}")
    print(f"input_idsのshape: {inputs['input_ids'].shape}")
    print(f"トークンID: {input_ids}")
    print(f"内部表記のトークン: {raw_tokens}")
    print(f"デコードしたトークン: {decoded_tokens}")

    # 入力データをモデルと同じデバイスへ移動する
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------
    # 5. GPT-2にプロンプトを入力する
    # --------------------------------------------------
    # 今回は学習を行わないため、勾配を計算しない
    # これにより、メモリ使用量を抑えられる
    with torch.no_grad():
        outputs = model(**inputs)

    # logitsのshapeは、
    # [バッチサイズ, トークン数, 語彙数]
    logits = outputs.logits

    print("\n===== モデルの出力 =====")
    print(f"logitsのshape: {logits.shape}")

    # --------------------------------------------------
    # 6. 次トークンの予測スコアを取り出す
    # --------------------------------------------------
    # logits[0, -1, :] の意味
    #
    # 0  : バッチの1番目
    # -1 : 入力系列の最後の位置
    # :  : 語彙に含まれる全トークン
    #
    # つまり、
    # 「The movie was full of」の次に来る
    # 各トークンのスコアを取り出している
    next_token_logits = logits[0, -1, :]

    # --------------------------------------------------
    # 7. スコアを確率に変換する
    # --------------------------------------------------
    # logitsは、そのままでは確率ではない
    # softmaxを適用すると、全候補の合計が1になる確率に変換される
    next_token_probabilities = torch.softmax(
        next_token_logits,
        dim=-1,
    )

    # --------------------------------------------------
    # 8. 確率が高い上位10トークンを取得する
    # --------------------------------------------------
    top_probabilities, top_token_ids = torch.topk(
        next_token_probabilities,
        k=TOP_K,
    )

    # --------------------------------------------------
    # 9. 結果を表示する
    # --------------------------------------------------
    print("\n===== 次トークンの予測結果 =====")

    for rank in range(TOP_K):
        # Tensorから通常のPythonの数値を取り出す
        token_id = top_token_ids[rank].item()
        probability = top_probabilities[rank].item()

        # GPT-2内部で使われるトークン表記
        raw_token = tokenizer.convert_ids_to_tokens(token_id)

        # 人間が読みやすい文字列へ変換する
        decoded_token = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
        )

        print(
            f"{rank + 1:2d}位: "
            f"token={decoded_token!r}, "
            f"raw_token={raw_token!r}, "
            f"token_id={token_id:5d}, "
            f"確率={probability:.6f} "
            f"({probability * 100:.3f}%)"
        )


if __name__ == "__main__":
    main()

#出力
# ===== プロンプトのトークン化結果 =====
# プロンプト: 'The movie was full of'
# input_idsのshape: torch.Size([1, 5])
# トークンID: [464, 3807, 373, 1336, 286]
# 内部表記のトークン: ['The', 'Ġmovie', 'Ġwas', 'Ġfull', 'Ġof']
# デコードしたトークン: ['The', ' movie', ' was', ' full', ' of']

# ===== モデルの出力 =====
# logitsのshape: torch.Size([1, 5, 50257])

# ===== 次トークンの予測結果 =====
#  1位: token=' great', raw_token='Ġgreat', token_id= 1049, 確率=0.023094 (2.309%)
#  2位: token=' references', raw_token='Ġreferences', token_id=10288, 確率=0.013512 (1.351%)
#  3位: token=' action', raw_token='Ġaction', token_id= 2223, 確率=0.013043 (1.304%)
#  4位: token=' moments', raw_token='Ġmoments', token_id= 7188, 確率=0.012449 (1.245%)
#  5位: token=' the', raw_token='Ġthe', token_id=  262, 確率=0.011860 (1.186%)
#  6位: token=' characters', raw_token='Ġcharacters', token_id= 3435, 確率=0.008720 (0.872%)
#  7位: token=' these', raw_token='Ġthese', token_id=  777, 確率=0.007216 (0.722%)
#  8位: token=' surprises', raw_token='Ġsurprises', token_id=24072, 確率=0.006894 (0.689%)
#  9位: token=' fun', raw_token='Ġfun', token_id= 1257, 確率=0.006526 (0.653%)
# 10位: token=' them', raw_token='Ġthem', token_id=  606, 確率=0.006154 (0.615%)