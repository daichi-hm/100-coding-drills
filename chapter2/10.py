#6.6コードの意図を書く p.76
#テキストを開き、行数をカウント
with open('popular-names.txt', 'r') as text:
    lines = text.readlines()
print("The number of lines:", len(lines))

#出力結果
#The number of lines: 2780

#unixコマンド（確認用）
#行数のカウント
#wc -l popular-names.txt