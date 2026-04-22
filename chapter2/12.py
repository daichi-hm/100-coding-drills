#6.6コードの意図を書く p.76
#テキストを開き、指定した行数をカウント
N=10 #ファイルの末尾から読み込む行数
with open('popular-names.txt', 'r') as text:
    lines = text.readlines()
    target_lines=lines[-10:]
    for line in target_lines:
        print(line.strip())

# 出力
# Liam    M       19837   2018
# Noah    M       18267   2018
# William M       14516   2018
# James   M       13525   2018
# Oliver  M       13389   2018
# Benjamin        M       13381   2018
# Elijah  M       12886   2018
# Lucas   M       12585   2018
# Mason   M       12435   2018
# Logan   M       12352   2018

#unix確認コマンド
#末尾から指定した行数を出力(デフォルトでは10行)
#tail popular-names.txt