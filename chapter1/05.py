#6.6コードの意図を書く p.76
#n-gram(単語単位）
def words_n_gram(sentence, num):
    sentence=sentence.split()
    result=[]
    lastWord=len(sentence)-num+1

    for i in range(lastWord):
        result.append(sentence[i:i+num])

    return result

#n-gram(文字単位)
def char_n_gram(sentence, num):
    sentence=sentence.replace(" ", "")
    result=[]
    lastWord=len(sentence)-num+1

    for i in range(lastWord):
        result.append(sentence[i:i+num])

    return result

sentence="I am an NLPer"
print(words_n_gram(sentence, 2))#
print(char_n_gram(sentence, 3))