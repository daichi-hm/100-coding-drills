text="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

#6.6コードの意図を書く p.76
#1文字だけ取り出したい単語の位置
keys=[1, 5, 6, 7, 8, 9, 15, 16, 19]

#テキストの不要な文字を除去し、スペースで分割する
SplitedText=text.replace(",", "").replace(".", "").split()

result={}

#keysの番号と一致したら、一文字取り出し、そうでない場合は2文字取り出す
for i, ExtractedWord, in enumerate(SplitedText, 1):
    if(i in keys):
        result[i]=ExtractedWord[0]
    else:
        result[i]=ExtractedWord[0:2]

print(result)
