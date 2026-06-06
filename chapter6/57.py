from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

# Google Newsで学習済みの単語ベクトルファイル
file_path = "GoogleNews-vectors-negative300.bin.gz"

# 単語アナロジーの評価データ
# 国名を取り出すために使う
analogy_file = "questions-words.txt"

# word2vec形式の学習済み単語ベクトルを読み込む
model = KeyedVectors.load_word2vec_format(file_path, binary=True)

# 国名を重複なしで保存するための集合
countries = set()

# 国名を取り出したいセクション
# どちらも「首都と国」の関係を扱うセクション
target_sections = {
    "capital-common-countries",
    "capital-world",
}

# 現在読んでいるセクション名を保存する変数
current_section = ""

# questions-words.txt を読み込む
with open(analogy_file, "r", encoding="utf-8") as f:

    # ファイルを1行ずつ読む
    for line in f:
        # 行末の改行を削除する
        line = line.strip()

        # セクション名の行は ":" で始まる
        if line.startswith(":"):
            # 例: ": capital-common-countries" から
            # "capital-common-countries" だけを取り出す
            current_section = line[2:]
            continue

        # 対象セクション以外は処理しない
        if current_section not in target_sections:
            continue

        # 1行を4つの単語に分ける
        # 例: Athens Greece Baghdad Iraq
        word1, word2, word3, word4 = line.split()

        # capital系セクションでは、
        # 2列目と4列目が国名
        countries.add(word2)
        countries.add(word4)

# 国名と、その国名の単語ベクトルを入れるリスト
country_names = []
country_vectors = []

# 集めた国名から、単語ベクトルが存在するものだけを取り出す
for country in countries:
    if country in model:
        country_names.append(country)
        country_vectors.append(model[country])

# k-meansクラスタリングを実行する
# n_clusters=5 は、5つのクラスタに分けるという意味
# n_init=10 は、初期値を変えて10回試し、その中でよい結果を使うという意味
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)

# 国名ベクトルを5つのクラスタに分類する
labels = kmeans.fit_predict(country_vectors)

# クラスタ番号ごとに国名をまとめる
clusters = {}

for country, label in zip(country_names, labels):
    if label not in clusters:
        clusters[label] = []

    clusters[label].append(country)

# 結果を表示する
for label in sorted(clusters):
    print(f"クラスタ {label}:")

    # 見やすいように国名をアルファベット順に並べる
    for country in sorted(clusters[label]):
        print(country)

    print()

# 出力
# クラスタ 0:
# Albania
# Armenia
# Azerbaijan
# Belarus
# Bulgaria
# Croatia
# Cyprus
# Estonia
# Georgia
# Greece
# Kazakhstan
# Kyrgyzstan
# Latvia
# Lithuania
# Macedonia
# Moldova
# Montenegro
# Romania
# Russia
# Serbia
# Slovakia
# Slovenia
# Tajikistan
# Turkey
# Turkmenistan
# Ukraine
# Uzbekistan

# クラスタ 1:
# Angola
# Botswana
# Burundi
# Eritrea
# Gabon
# Gambia
# Ghana
# Guinea
# Kenya
# Liberia
# Madagascar
# Malawi
# Mali
# Mozambique
# Namibia
# Niger
# Nigeria
# Rwanda
# Senegal
# Uganda
# Zambia
# Zimbabwe

# クラスタ 2:
# Australia
# Austria
# Belgium
# Canada
# China
# Denmark
# England
# Finland
# France
# Germany
# Greenland
# Hungary
# Ireland
# Italy
# Japan
# Liechtenstein
# Malta
# Norway
# Poland
# Portugal
# Spain
# Sweden
# Switzerland

# クラスタ 3:
# Bahamas
# Bangladesh
# Belize
# Bhutan
# Chile
# Cuba
# Dominica
# Ecuador
# Fiji
# Guyana
# Honduras
# Indonesia
# Jamaica
# Laos
# Nepal
# Nicaragua
# Peru
# Philippines
# Samoa
# Suriname
# Taiwan
# Thailand
# Tuvalu
# Uruguay
# Venezuela
# Vietnam

# クラスタ 4:
# Afghanistan
# Algeria
# Bahrain
# Egypt
# Iran
# Iraq
# Jordan
# Lebanon
# Libya
# Mauritania
# Morocco
# Oman
# Pakistan
# Qatar
# Somalia
# Sudan
# Syria
# Tunisia
