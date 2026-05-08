#6.6コードの意図を書く p.76
#テキストを開き、指定した行数をカウント
N=10 #ファイルの先頭から読み込む行数
with open('popular-names.txt', 'r') as text:
    lines = text.readlines()
    target_lines=lines[0:N]
    for line in target_lines:
        print(line.replace("\t", " ").strip())#タブ1文字をスペース1文字に置換

#出力
# Mary F 7065 1880
# Anna F 2604 1880
# Emma F 2003 1880
# Elizabeth F 1939 1880
# Minnie F 1746 1880
# Margaret F 1578 1880
# Ida F 1472 1880
# Alice F 1414 1880
# Bertha F 1320 1880
# Sarah F 1288 1880

#unix確認コマンド
#先頭から10行を出力し、タブ1文字につきスペース1文字に置換
#head -n 10 popular-names.txt | sed 's/\t/ /g'