text="stressed"
ansText=""
textLength=len(text)

#6.6コードの意図を書く p.76
#テキストを逆順で出力
for i in range(textLength-1,-1,-1):
    ansText+=text[i]

print(ansText)