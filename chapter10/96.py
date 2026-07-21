import csv
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "openai-community/gpt2-medium"


# --------------------------------------------------
# データファイルの場所
# --------------------------------------------------
# __file__
# → 現在実行している96.pyの場所
#
# Path(__file__).resolve().parent
# → chapter10ディレクトリ
SCRIPT_DIR = Path(__file__).resolve().parent

DEV_FILE = (
    SCRIPT_DIR.parent
    / "chapter7"
    / "SST-2"
    / "dev.tsv"
)


def load_sst2_dev(file_path):
    """
    SST-2のdev.tsvを読み込み、
    文章と正解ラベルを辞書のリストとして返す関数。
    """

    # データを保存するリスト
    examples = []

    # TSVファイルを開く
    #
    # TSVは、値がタブで区切られたファイル
    with file_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as file:
        # delimiter="\t"により、
        # タブ区切りのファイルとして読み込む
        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        # 1行ずつ読み込む
        for row in reader:
            # dev.tsvの各行には、
            #
            # sentence : レビュー文
            # label    : 正解ラベル
            #
            # が入っている
            sentence = row["sentence"]

            # ファイルから読んだ値は文字列なので、
            # int()で整数へ変換する
            label = int(row["label"])

            examples.append(
                {
                    "sentence": sentence,
                    "label": label,
                }
            )

    return examples


def build_prompt(sentence):
    """
    1つのレビュー文から、
    GPT-2へ与える感情分析用プロンプトを作る関数。
    """

    # GPT-2は指示に従うことだけを目的に
    # 学習されたモデルではないため、
    # positiveとnegativeの例を1つずつ示す
    #
    # 最後のSentiment:より後ろを
    # GPT-2に予測させる
    prompt = (
        "Classify the sentiment of the movie review "
        "as positive or negative.\n\n"
        "Review: I loved this movie.\n"
        "Sentiment: positive\n\n"
        "Review: This movie was boring and terrible.\n"
        "Sentiment: negative\n\n"
        f"Review: {sentence}\n"
        "Sentiment:"
    )

    return prompt


def get_label_token_id(tokenizer, label_word):
    """
    positiveまたはnegativeの文字列を
    GPT-2のトークンIDへ変換する関数。
    """

    # プロンプトは"Sentiment:"で終わっているため、
    # その後ろには空白付きで、
    #
    # " positive"
    # " negative"
    #
    # が続く想定にする
    text = " " + label_word

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    # 今回は、positiveとnegativeが
    # それぞれ1トークンになることを利用する
    if len(token_ids) != 1:
        raise ValueError(
            f"{text!r}が1トークンではありません。"
            f"token_ids={token_ids}"
        )

    return token_ids[0]


def predict_sentiment(
    sentence,
    model,
    tokenizer,
    device,
    negative_token_id,
    positive_token_id,
):
    """
    1つのレビュー文について、
    negativeまたはpositiveを予測する関数。

    戻り値：
    predicted_label
        0ならnegative
        1ならpositive

    negative_probability
        2候補の中でのnegativeの相対確率

    positive_probability
        2候補の中でのpositiveの相対確率
    """

    # --------------------------------------------------
    # 1. 感情分析用のプロンプトを作る
    # --------------------------------------------------
    prompt = build_prompt(sentence)

    # --------------------------------------------------
    # 2. プロンプトをトークン化する
    # --------------------------------------------------
    inputs = tokenizer(
        prompt,
        return_tensors="pt",

        # 非常に長いレビューがあった場合は、
        # 最大256トークンまでに切る
        truncation=True,
        max_length=256,
    )

    # input_idsやattention_maskを、
    # モデルと同じGPUまたはCPUへ移動する
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------
    # 3. GPT-2から全位置の予測スコアを取得する
    # --------------------------------------------------
    # 今回は学習しないため、勾配を計算しない
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.logitsのshapeは、
    #
    # [バッチサイズ, トークン数, 語彙数]
    #
    # 今回は、
    # [1, プロンプトのトークン数, 50257]
    # のようになる
    logits = outputs.logits

    # --------------------------------------------------
    # 4. 最後の位置の全語彙スコアを取り出す
    # --------------------------------------------------
    # [0]
    # → バッチ内の0番目の文章
    #
    # [-1]
    # → プロンプトの最後の位置
    #
    # [:]
    # → 全語彙候補
    #
    # プロンプトは"Sentiment:"で終わっているので、
    # その次に何が来るかのスコアになる
    next_token_logits = logits[0, -1, :]

    # --------------------------------------------------
    # 5. negativeとpositiveのスコアだけを取り出す
    # --------------------------------------------------
    negative_logit = next_token_logits[
        negative_token_id
    ]

    positive_logit = next_token_logits[
        positive_token_id
    ]

    # 2つのスコアを1つのTensorにまとめる
    #
    # 0番目：negative
    # 1番目：positive
    candidate_logits = torch.stack(
        [
            negative_logit,
            positive_logit,
        ]
    )

    # --------------------------------------------------
    # 6. 2候補の相対確率へ変換する
    # --------------------------------------------------
    # negativeとpositiveだけを候補とした場合に、
    # 合計が1になる確率へ変換する
    candidate_probabilities = torch.softmax(
        candidate_logits,
        dim=0,
    )

    # 0番目がnegativeの確率
    negative_probability = (
        candidate_probabilities[0].item()
    )

    # 1番目がpositiveの確率
    positive_probability = (
        candidate_probabilities[1].item()
    )

    # --------------------------------------------------
    # 7. スコアが高かった方を予測ラベルにする
    # --------------------------------------------------
    # argmax()は最大値の位置を返す
    #
    # 0が返る
    # → negativeの方が高い
    #
    # 1が返る
    # → positiveの方が高い
    predicted_label = torch.argmax(
        candidate_logits
    ).item()

    return (
        predicted_label,
        negative_probability,
        positive_probability,
        prompt,
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
    # 2. SST-2のdev.tsvが存在するか確認する
    # --------------------------------------------------
    if not DEV_FILE.exists():
        raise FileNotFoundError(
            "SST-2のdev.tsvが見つかりません。\n"
            f"探した場所: {DEV_FILE}"
        )

    # --------------------------------------------------
    # 3. 検証データを読み込む
    # --------------------------------------------------
    dev_examples = load_sst2_dev(DEV_FILE)

    print(f"検証データ数: {len(dev_examples)}")

    # --------------------------------------------------
    # 4. トークナイザとモデルを読み込む
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
    # 5. ラベル語のトークンIDを取得する
    # --------------------------------------------------
    negative_token_id = get_label_token_id(
        tokenizer,
        "negative",
    )

    positive_token_id = get_label_token_id(
        tokenizer,
        "positive",
    )

    print(
        f"' negative'のトークンID: "
        f"{negative_token_id}"
    )

    print(
        f"' positive'のトークンID: "
        f"{positive_token_id}"
    )

    # --------------------------------------------------
    # 6. dev.tsvの全レビューを分類する
    # --------------------------------------------------
    correct_count = 0

    print("\n===== 最初の5件の予測結果 =====")

    for index, example in enumerate(dev_examples):
        sentence = example["sentence"]
        gold_label = example["label"]

        (
            predicted_label,
            negative_probability,
            positive_probability,
            prompt,
        ) = predict_sentiment(
            sentence=sentence,
            model=model,
            tokenizer=tokenizer,
            device=device,
            negative_token_id=negative_token_id,
            positive_token_id=positive_token_id,
        )

        # 予測と正解が同じなら、
        # 正解数を1増やす
        if predicted_label == gold_label:
            correct_count += 1

        # 最初の5件だけ、詳しく表示する
        if index < 5:
            gold_name = (
                "positive"
                if gold_label == 1
                else "negative"
            )

            predicted_name = (
                "positive"
                if predicted_label == 1
                else "negative"
            )

            print(f"\n事例{index + 1}")
            print(f"文章      : {sentence}")
            print(
                f"正解      : "
                f"{gold_label} ({gold_name})"
            )
            print(
                f"予測      : "
                f"{predicted_label} "
                f"({predicted_name})"
            )
            print(
                f"negative  : "
                f"{negative_probability:.4f}"
            )
            print(
                f"positive  : "
                f"{positive_probability:.4f}"
            )

    # --------------------------------------------------
    # 7. 正解率を計算する
    # --------------------------------------------------
    total_count = len(dev_examples)

    accuracy = correct_count / total_count

    print("\n===== 評価結果 =====")
    print(f"正解数  : {correct_count}")
    print(f"全事例数: {total_count}")
    print(f"正解率  : {accuracy:.4f}")
    print(f"正解率  : {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

#出力
# 使用するデバイス: cuda
# 検証データ数: 872
# ' negative'のトークンID: 4633
# ' positive'のトークンID: 3967

# ===== 最初の5件の予測結果 =====

# 事例1
# 文章      : it 's a charming and often affecting journey . 
# 正解      : 1 (positive)
# 予測      : 1 (positive)
# negative  : 0.4297
# positive  : 0.5703

# 事例2
# 文章      : unflinchingly bleak and desperate 
# 正解      : 0 (negative)
# 予測      : 1 (positive)
# negative  : 0.3976
# positive  : 0.6024

# 事例3
# 文章      : allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker . 
# 正解      : 1 (positive)
# 予測      : 0 (negative)
# negative  : 0.5221
# positive  : 0.4779

# 事例4
# 文章      : the acting , costumes , music , cinematography and sound are all astounding given the production 's austere locales . 
# 正解      : 1 (positive)
# 予測      : 0 (negative)
# negative  : 0.7869
# positive  : 0.2131

# 事例5
# 文章      : it 's slow -- very , very slow . 
# 正解      : 0 (negative)
# 予測      : 0 (negative)
# negative  : 0.7199
# positive  : 0.2801

# ===== 評価結果 =====
# 正解数  : 567
# 全事例数: 872
# 正解率  : 0.6502
# 正解率  : 65.02%