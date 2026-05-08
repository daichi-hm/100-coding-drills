#6.6コードの意図を書く p.76
#テキストを開き、指定した行数をカウント
N=10 #ファイルの先頭から読み込む行数
with open('popular-names.txt', 'r') as text:
    lines = text.readlines()
    target_lines = lines[0:N]
    #1列目だけ取り出す
    for line in target_lines:
        columns = line.strip().split()
        print(columns[0])
        
#出力
# Mary
# Anna
# Emma
# Elizabeth
# Minnie
# Margaret
# Ida
# Alice
# Bertha
# Sarah

#unix確認コマンド
#先頭から10行を出力し、1列目だけを取り出す
#head -n 10 popular-names.txt | cut -f 1