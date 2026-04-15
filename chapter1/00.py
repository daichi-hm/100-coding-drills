text1="パトカー"
text2="タクシー"

ansText=""

#6.6コードの意図を書く p.76
#2つの文字列の文字を先頭から交互に連結する
for i in range(len(text1)):
    ansText+=text1[i]
    ansText+=text2[i]

print(ansText)