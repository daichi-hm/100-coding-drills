#6.6コードの意図を書く p.76
#暗号化、または復号化する関数（英語の小文字のみを暗号化）
def cipher(text):
    result=""
    for c in text:
        if c.islower():
            result += chr(219-ord(c))#ordで文字から文字コードに変換
        else:
            result += c

    return result

#暗号化
encrypted = cipher("THIS is Test.")
print(encrypted)

#復号化
decrypted = cipher(encrypted)
print(decrypted)