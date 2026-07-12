from pathlib import Path

import torch
from transformers import BertTokenizer, BertForSequenceClassification


# 87番と同じモデル名を使う
# 本来の指定は "google-bert/bert-base-uncased" だが、
MODEL_NAME = "bert-base-uncased"

# 最大系列長
# 87番と同じ設定にしておく
MAX_LENGTH = 128

# 現在のファイル 88.py があるディレクトリ
CURRENT_DIR = Path(__file__).resolve().parent

# 87番で保存したbestモデルのパス
CHECKPOINT_PATH = CURRENT_DIR / "best_bert_sst2.pt"


def predict_sentiment(model, tokenizer, sentences, device):
    """文のリストを受け取り、極性ラベルと確率を予測する関数"""

    # 文をBERTに入力できる形式へ変換する
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # 入力TensorをGPUまたはCPUへ送る
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 評価モードにする
    # Dropoutなどを無効にして、予測用の挙動にする
    model.eval()

    # 予測時は勾配計算をしない
    with torch.no_grad():
        outputs = model(**inputs)

    # logitsは、各文に対する分類スコア
    # shape: [文の数, クラス数]
    # SST-2ではクラス数は2なので、shapeは [5, 2] になる
    logits = outputs.logits

    # logitsを確率に変換する
    # dim=1 は「各文ごとに、negative/positive方向でsoftmaxする」という意味
    probabilities = torch.softmax(logits, dim=1)

    # 確率が一番高いクラスを予測ラベルとする
    predictions = torch.argmax(probabilities, dim=1)

    return predictions, probabilities


def main():
    # GPUが使えるならGPUを使う
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 87番の保存済みモデルが存在するか確認する
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT_PATH} が見つかりません。先に87.pyを実行して、best_bert_sst2.ptを作成してください。"
        )

    # tokenizerを読み込む
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 87番と同じ構造のモデルを作る
    # num_labels=2 は negative / positive の2クラス分類という意味
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # 87番で保存したcheckpointを読み込む
    # map_location=device により、保存時のGPUに依存せず、
    # 今使っているGPUまたはCPUに読み込める
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    # checkpoint内の学習済み重みをmodelに読み込む
    model.load_state_dict(checkpoint["model_state_dict"])

    # モデルをGPUまたはCPUへ送る
    model.to(device)

    # 87番で保存したbest情報を表示する
    print("loaded checkpoint:")
    print("best epoch:", checkpoint["best_epoch"])
    print(f"best valid accuracy: {checkpoint['best_valid_accuracy']:.6f}")
    print(f"best valid loss: {checkpoint['best_valid_loss']:.6f}")

    # 問題88で指定されている5文
    sentences = [
        "The movie was full of incomprehensibilities.",
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish.",
    ]

    # 予測する
    predictions, probabilities = predict_sentiment(
        model,
        tokenizer,
        sentences,
        device,
    )

    # ラベルIDを人間が読める名前に変換するための辞書
    # SST-2では 0 が negative, 1 が positive
    id_to_label = {
        0: "negative",
        1: "positive",
    }

    print("\npredictions:")
    for sentence, prediction, probability in zip(sentences, predictions, probabilities):
        predicted_id = prediction.item()
        predicted_label = id_to_label[predicted_id]

        negative_prob = probability[0].item()
        positive_prob = probability[1].item()

        print("\nsentence:")
        print(sentence)

        print("prediction:")
        print(f"{predicted_label} ({predicted_id})")

        print("probability:")
        print(f"negative: {negative_prob:.6f}")
        print(f"positive: {positive_prob:.6f}")


if __name__ == "__main__":
    main()

#出力
# device: cuda
# loaded checkpoint:
# best epoch: 1
# best valid accuracy: 0.920872
# best valid loss: 0.236477

# predictions:

# sentence:
# The movie was full of incomprehensibilities.
# prediction:
# negative (0)
# probability:
# negative: 0.996050
# positive: 0.003950

# sentence:
# The movie was full of fun.
# prediction:
# positive (1)
# probability:
# negative: 0.001202
# positive: 0.998798

# sentence:
# The movie was full of excitement.
# prediction:
# positive (1)
# probability:
# negative: 0.003553
# positive: 0.996447

# sentence:
# The movie was full of crap.
# prediction:
# negative (0)
# probability:
# negative: 0.997912
# positive: 0.002088

# sentence:
# The movie was full of rubbish.
# prediction:
# negative (0)
# probability:
# negative: 0.997997
# positive: 0.002003