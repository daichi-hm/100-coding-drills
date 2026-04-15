#6.6コードの意図を書く p.76
#n-gram(文字単位)
def char_n_gram(sentence, num):
    sentence=sentence.replace(" ", "")
    result=[]
    lastWord=len(sentence)-num+1

    for i in range(lastWord):
        result.append(sentence[i:i+num])

    return result

text1="paraparaparadise"
text2="paragraph"

setX=set(char_n_gram(text1, 2))
setY=set(char_n_gram(text2, 2))

print("set X:", setX)
print("set Y:", setY)


X_or_Y = setX | setY
print("Union:", X_or_Y)

X_and_Y = setX & setY
print("Intersection:", X_and_Y)

X_minus_Y = setX - setY
print("Difference:", X_minus_Y)

#seが集合X, Yにあるか確認する
if "se" in setX:
    print("\"se\" belongs to set X")
else:
    print("\"se\" doesn't belong to set X")

if "se" in setY:
    print("\"se\" belongs to set Y")
else:
    print("\"se\" doesn't belong to set Y")

