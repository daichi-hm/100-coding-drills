text="パタトクカシーー"
ansText=""

#6.6コードの意図を書く p.76
#文字列の2,4,6,8文字目を取り出し連結する
for i in range(1, 8, 2):
    ansText+=text[i]

print(ansText)