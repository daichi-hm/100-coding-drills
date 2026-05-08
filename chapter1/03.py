text="Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

#6.6コードの意図を書く p.76
#不要な文字を除去して、スペースで単語を分割する
SplitedText=text.replace(",", "").replace(".", "").split(" ")

#各単語の文字数を出す
for i in SplitedText:
    print(len(i))