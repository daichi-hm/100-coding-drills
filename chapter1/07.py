#6.6コードの意図を書く p.76
#引数を受け取り、テンプレート文を返す関数
def text_template(x, y, z):
    sentence="{hour}の時の{temperature}は{value}"
    return sentence.format(hour=x, temperature=y, value=z)

print(text_template(12, "気温", 22.4))