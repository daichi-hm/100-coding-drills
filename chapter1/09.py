#6.6コードの意図を書く p.76
import random#shuffleを使うために

text="I couldn\'t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

#スペースで文を単語ごとに分割する
SplitedText = text.split()

print(SplitedText)

result = []

#分割された単語が4文字以下の時は何もしない
#5文字以上の時は、単語の先頭文字と終端文字以外のすべての文字をランダムで並べ替える
for word in SplitedText:
    if len(word) <= 4:
        result.append(word)

    else:
        sentence = list(word[1:-1])#先頭文字と終端文字の間の文字をリスト化
        random.shuffle(sentence)
        sentence = word[0] + "".join(sentence) + word[-1]#並べ替えるためにリスト化したものを文字列に戻したうえで結合

        result.append(sentence)

print(result)
    