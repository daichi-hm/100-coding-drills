#6.6コードの意図を書く p.76
#1列目の文字列の種類の数をカウントする関数
def get_unique_first_column(input_file):
    #重複を自動で排除できるためにsetを使用
    unique_elements = set()

    with open(input_file, 'r') as text:
        for line in text:
            #行をタブで分割し、1列目を取得
            cols = line.split('\t')
            if cols:
                unique_elements.add(cols[0])

        #種類の合計を表示
        print("Type:",len(unique_elements))

get_unique_first_column('popular-names.txt')

#出力
#Type: 136

#確認用unixコマンド
#cut -f 1 popular-names.txt | sort | uniq | wc -l