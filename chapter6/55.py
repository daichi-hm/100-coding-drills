# 54番で作成した結果ファイル
input_file = "answer_54.txt"

# 正解した問題数を数える変数
correct_count = 0

# 全問題数を数える変数
total_count = 0

# 54番の結果ファイルを読み込み用に開く
with open(input_file, "r", encoding="utf-8") as f:

    # ファイルを1行ずつ読む
    for line in f:
        # 行末の改行を削除する
        line = line.strip()

        # 空行があった場合は飛ばす
        if line == "":
            continue

        # 行を空白で分割する
        # 例:
        # Athens Greece Bangkok Thailand Thailand 0.7137669324874878
        cols = line.split()

        # 4列目が正解単語
        # Pythonでは0番目から数えるので、4列目は cols[3]
        correct_word = cols[3]

        # 5列目がモデルの予測単語
        # Pythonでは0番目から数えるので、5列目は cols[4]
        predicted_word = cols[4]

        # 全問題数を1増やす
        total_count += 1

        # 正解単語と予測単語が一致していれば、正解数を1増やす
        if correct_word == predicted_word:
            correct_count += 1

# 正解率を計算する
accuracy = correct_count / total_count

# 結果を表示する
print("意味的アナロジーの正解数:", correct_count)
print("意味的アナロジーの全問題数:", total_count)
print(f"意味的アナロジーの正解率: {accuracy * 100:.2f}%")

# 出力
# 意味的アナロジーの正解数: 421
# 意味的アナロジーの全問題数: 506
# 意味的アナロジーの正解率: 83.20%
