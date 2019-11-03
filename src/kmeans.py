import pandas as pd
import numpy as np

# read file and rename column
dataset = pd.read_csv('/home/knight/Documents/TA/Data After OpenRefine/shoesman-json.tsv', sep="\t",
                      encoding="ISO-8859-1")
dataset = dataset.rename(index=str, columns={"doc - product": "nama", "doc - jenis": "jenis", "doc - rating": "rating",
                                             "doc - tanggal": "tanggal", "doc - nama": "nama-orang",
                                             "doc - isireview": "review"})

# remove duplicate and delete column
dataset = dataset.drop_duplicates()

import sqlalchemy

# sum of empty data
print(dataset['nama'].isnull().sum())
print(dataset['tanggal'].isnull().sum())

# drop rows with empty data
dataset = dataset.dropna(axis=0, subset=['nama'])
dataset = dataset.dropna(axis=0, subset=['tanggal'])

dataset = dataset.reset_index(drop=True)

# convert text to lowercase
dataset['nama'] = dataset['nama'].str.lower()

# remove punctuation
import string

for x, y in enumerate(dataset['nama']):
    dataset['nama'][x] = dataset['nama'][x].translate(str.maketrans("", "", string.punctuation))

# remove word with number
import re

dataset['nama'] = dataset['nama'].str.replace(r'\w*\d\w*'.format("|".join([re.escape(x) for x in dataset['nama']])),
                                              " ")

# remove white space
for x, y in enumerate(dataset['nama']):
    dataset['nama'][x] = re.sub(' +', ' ', dataset['nama'][x])

# remove word with accent
import unidecode

for i, j in enumerate(dataset['nama']):
    dataset['nama'][i] = unidecode.unidecode(dataset['nama'][i])

# remove stop words
from nltk.corpus import stopwords

stop = stopwords.words('english')
dataset['nama'] = dataset['nama'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# stemming
"""" from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

dataset['nama'] = dataset['nama'].apply(stem_sentences)"""

# lemmatization
from nltk.stem.snowball import SnowballStemmer

englishStemmer = SnowballStemmer("english")
for i, j in enumerate(dataset['nama']):
    dataset['nama'][i] = englishStemmer.stem(dataset['nama'][i])

# named entity recognition
"""import spacy
from spacy import displacy
import en_core_web_sm
nlp = spacy.load("en")

text = [[1, 'University of California has great research located in San Diego'],[2, 'MIT is at Boston']]
df = pd.DataFrame(text, columns = ['id', 'text'])
df['new_text'] = df['text'].apply(lambda x: list(nlp(x).ents))
print(df["text"])"""

# remove word that not in tag
import nltk

for x, y in enumerate(dataset['nama']):
    tokens = nltk.word_tokenize(dataset['nama'][x])
    tagged = nltk.pos_tag(tokens)
    dataset['nama'][x] = [word for word, tag in tagged if
                          tag == 'IN' or tag == 'RB' or tag == 'RBR' or tag == 'RBS' or tag == 'RP' or tag == 'VB' or tag == 'VBP' or tag == 'JJ' or tag == 'VBN' or tag == 'NNP' or tag == 'NNS' or tag == 'NNPS' or tag == 'NN' or tag == 'FW']
    dataset['nama'][x] = ' '.join(dataset['nama'][x])

# extract all word
'''listword = []
for x in range(len(dataset['nama'])) :
    tokens = nltk.word_tokenize(dataset['nama'][x])
    listword += tokens

listword = list(dict.fromkeys(listword))
for x in listword :
    print(x)'''

# preprocess replace from extract word that already reviewed
word = pd.read_csv('/home/knight/Documents/Book1.csv', encoding='cp1252', names=["words"])

test = word['words'].values.tolist()
data_dict = dict.fromkeys(test, "")
for x, y in enumerate(dataset['nama']):
    for j in data_dict:
        dataset['nama'][x] = re.sub(r"\b" + re.escape(j) + r"\b", "", dataset['nama'][x])
    dataset['nama'][x] = re.sub(' +', ' ', dataset['nama'][x])
    dataset['nama'][x] = dataset['nama'][x].strip()
    print(dataset['nama'][x])
    if x % 3000 == 0:
        dataset.to_csv('/home/knight/Documents/preprocess.tsv', sep="\t")

# convert to tsv file
dataset.to_csv('/home/knight/Documents/preprocess.tsv', sep="\t")

for i, j in enumerate(dataset['nama']):
    if len(dataset['nama'][i]) == 0:
        dataset['nama'][i] = np.nan

# sum of empty data
print(dataset['nama'].isnull().sum())
print(dataset['tanggal'].isnull().sum())

# drop rows with empty data
dataset = dataset.dropna(axis=0, subset=['nama'])
dataset = dataset.dropna(axis=0, subset=['tanggal'])

dataset = dataset.reset_index(drop=True)

# 0 tokenize
nama_token = []
for x, y in enumerate(dataset['nama']):
    nama_token.append(dataset['nama'][x].split())

# 1 bag of word
listword = []
for x, y in enumerate(dataset['nama']):
    tokens = nltk.word_tokenize(dataset['nama'][x])
    listword += tokens

worddict = dict.fromkeys(listword, 0)

# 2 create data with 0 value same with sum of data
word_token = []
for x in dataset['nama']:
    word_token.append(dict.fromkeys(worddict, 0))

# 3 add = 1 for word same in nama_token
for x, y in enumerate(nama_token):
    for word in nama_token[x]:
        word_token[x][word] += 1

# 4 tf function
'''def compute_tf(word_dict, l):
    tf = {}
    sum_nk = len(l)
    for word, count in word_dict.items():
        tf[word] = count / sum_nk
    return tf'''

# 5 running tf from data before
tf = []
for x, y in enumerate(word_token):
    # tf.append(compute_tf(word_token[x],nama_token[x]))
    tf.append(word_token[x].values())

# data feature extraction save as dataframe
readyclustering = pd.DataFrame(data=tf)
data = pd.concat([dataset['nama'], dataset['tanggal'], readyclustering], axis=1)
data.to_csv('/home/knight/Documents/tf.tsv', sep="\t", index=False, index_label=False)

# predict k with elbow
'''from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
Sum_of_squared_distances = []
K = range(10,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(readyclustering)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()'''

# k means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=12, max_iter=1000).fit(readyclustering)
clusters = kmeans.fit_predict(readyclustering)
score = kmeans.score(readyclustering)

hasilcluster = []
for i in clusters:
    hasilcluster.append(i)

clusterresult = pd.DataFrame({'Assignment': hasilcluster})
data = pd.read_csv('/home/knight/Documents/tf.tsv', sep="\t")

datacluster = pd.concat([clusterresult, data['nama']], axis=1)
datacluster.to_csv('/home/knight/Documents/kmeans.tsv', sep="\t", index=False, index_label=False)

# counting

data = pd.read_csv('/home/knight/Documents/kmeans.tsv', sep="\t")
data = data.sort_values(by=['Assignment'])
data = data.reset_index(drop=True)

# Menjumlah tiap assignment
a = 0
jumlah = [1]
assignment = data['Assignment'][0]
for i, j in enumerate(data['Assignment']):
    if i != 0:
        if data['Assignment'][i] == assignment:
            jumlah[a] += 1
        else:
            a += 1
            jumlah.append(1)
            assignment = data['Assignment'][i]

# Mencari nama produk
nama_token = []
listword = []
produk = []
assignment = data['Assignment'][0]
for i, j in enumerate(data['Assignment']):
    nama_token.append(data['nama'][i].split())
    tokens = nltk.word_tokenize(data['nama'][i])
    listword += tokens

    if i != len(data['Assignment']) - 1 and data['Assignment'][i + 1] != data['Assignment'][i]:
        worddict = dict.fromkeys(listword, 0)
        for z, y in enumerate(nama_token):
            for word in nama_token[z]:
                worddict[word] += 1
        maxv = max(worddict.values())
        produk.append([k for k, v in worddict.items() if v == maxv])

        listword.clear()
        nama_token.clear()
    elif i == len(data['Assignment']) - 1:
        worddict = dict.fromkeys(listword, 0)
        for z, y in enumerate(nama_token):
            for word in nama_token[z]:
                worddict[word] += 1
        maxv = max(worddict.values())
        produk.append([k for k, v in worddict.items() if v == maxv])

        listword.clear()
        nama_token.clear()

for i, j in enumerate(produk):
    if not isinstance(produk[i], str):
        produk[i] = ' '.join(produk[i])
# convert to dataframe and sort
produkfix = pd.DataFrame({'Produk': produk, 'Jumlah': jumlah})
produkfix = produkfix.reset_index(drop=True)

produkfix = produkfix
produkfix = produkfix.sort_values(by=['Jumlah'], ascending=True)
produkfix2 = produkfix
produkfix3 = produkfix

for i, j in enumerate(produkfix['Produk']):
    if i >= ((len(produkfix['Produk']) / 2) + 10) or i <= ((len(produkfix['Produk']) / 2) - 20):
        produkfix2 = produkfix2.drop(i, axis=0)

for i, j in enumerate(produkfix['Produk']):
    if i <= (len(produkfix['Produk']) - 30):
        produkfix3 = produkfix3.drop(i, axis=0)

for i, j in enumerate(produkfix['Produk']):
    if i >= 30:
        produkfix = produkfix.drop(i, axis=0)

import matplotlib.pyplot as plt

plt.barh(produkfix['Produk'], produkfix['Jumlah'])
plt.ylabel('Produk Name')
plt.show()
# plt.savefig("C:/Users/Knight/Desktop/plot.png", dpi=100)

plt.barh(produkfix2['Produk'], produkfix2['Jumlah'])
plt.ylabel('Produk Name')
plt.show()
# plt.savefig("C:/Users/Knight/Desktop/plot.png", dpi=100)

plt.barh(produkfix3['Produk'], produkfix3['Jumlah'])
plt.ylabel('Produk Name')
plt.show()
# plt.savefig("C:/Users/Knight/Desktop/plot.png", dpi=100)

# group by date
from datetime import datetime

dataset = pd.read_csv('/home/knight/Documents/preprocess.tsv', sep="\t", encoding="ISO-8859-1")
data = pd.read_csv('/home/knight/Documents/kmeans.tsv', sep="\t")
dataset = pd.concat([dataset, data['Assignment']], axis=1)
del dataset['Unnamed: 0']
dataset = dataset.reset_index(drop=True)

for x, y in enumerate(dataset['tanggal']):
    if x == len(dataset['tanggal']):
        break
    try:
        dataset['tanggal'][x] = datetime.strptime(dataset['tanggal'][x], '%B %d, %Y')
    except ValueError:
        print(ValueError)
        if isinstance(dataset['tanggal'][x], datetime) is False:
            dataset = dataset.drop(x, axis=0)
            dataset = dataset.reset_index(drop=True)
            dataset['tanggal'][x] = datetime.strptime(dataset['tanggal'][x], '%B %d, %Y')

months = [g for n, g in dataset.set_index('tanggal').groupby(pd.Grouper(freq='M'))]

i = 0
while i <= len(months):
    if i == len(months):
        break
    if months[i].empty is True:
        del months[i]
        i = i - 1
    i = i + 1

# moga jalan
tanggallist = []
for h, j in enumerate(months):
    months[h] = months[h].sort_values(by=['Assignment'])
    tanggallist.append(months[h].index[0].strftime('%B %Y'))
    months[h] = months[h].reset_index(drop=False)

produkya = []
for h, r in enumerate(months):
    produk = []
    jumlah = []
    if len(months[h]) != 1:
        # Menjumlah tiap assignment
        a = 0
        jumlah = [1]
        assignment = months[h]['Assignment'][0]
        for i, j in enumerate(months[h]['Assignment']):
            if i != 0:
                if months[h]['Assignment'][i] == assignment:
                    jumlah[a] += 1
                else:
                    a += 1
                    jumlah.append(1)
                    assignment = months[h]['Assignment'][i]

        # Mencari nama produk
        nama_token = []
        listword = []
        produk = []
        assignment = months[h]['Assignment'][0]
        for i, j in enumerate(months[h]['Assignment']):
            nama_token.append(months[h]['nama'][i].split())
            tokens = nltk.word_tokenize(months[h]['nama]'][i])
            listword += tokens

            if i != len(months[h]['Assignment']) - 1 and months[h]['Assignment'][i + 1] != months[h]['Assignment'][i]:
                worddict = dict.fromkeys(listword, 0)
                for z, y in enumerate(nama_token):
                    for word in nama_token[z]:
                        worddict[word] += 1
                maxv = max(worddict.values())
                produk.append([k for k, v in worddict.items() if v == maxv])

                listword.clear()
                nama_token.clear()
            elif i == len(months[h]['Assignment']) - 1:
                worddict = dict.fromkeys(listword, 0)
                for z, y in enumerate(nama_token):
                    for word in nama_token[z]:
                        worddict[word] += 1
                maxv = max(worddict.values())
                produk.append([k for k, v in worddict.items() if v == maxv])

                listword.clear()
                nama_token.clear()

        for i, j in enumerate(produk):
            if not isinstance(produk[i], str):
                produk[i] = ' '.join(produk[i])
        # convert to dataframe and sort
    else:
        jumlah.append(1)
        produk.append(months[h]['nama'][0])

    produkfix = pd.DataFrame({'Produk': produk, 'Jumlah': jumlah})
    produkya.append(produkfix)
    del produkfix

for x, y in enumerate(produkya):
    produkya[x] = produkya[x].sort_values(by=['Jumlah'], ascending=True)
    plt.barh(produkya[x]['Produk'], produkya[x]['Jumlah'])
    plt.ylabel(tanggallist[x])
    plt.show()

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

newsgroups_train = fetch_20newsgroups(subset='train',
                                      categories=['alt.atheism', 'sci.space'])
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
])
X = pipeline.fit_transform(newsgroups_train.data).todense()

pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
plt.scatter(data2D[:, 0], data2D[:, 1], c=newsgroups_train.target)
plt.show()  # not required if using ipython notebook

import pyodbc
# insert to db
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=ecommerceta.database.windows.net;DATABASE=ecommerceta;UID=knight;PWD=Arief-1305')
cursor = conn.cursor()


for i, j in enumerate(dataset['nama-orang']):
    try:
        print(i)
        cursor.execute(
            "INSERT INTO dbo.ecommercetrends_sepatupria([nama],[jenis],[rating],[tanggal],[produk],[review]) values (?,?,?,?,?,?)",
            dataset['nama-orang'][i], dataset['jenis'][i], dataset['rating'][i], dataset['tanggal'][i], dataset['nama'][i],
            dataset['review'][i])
    except Exception as e:
        print(e)

conn.commit()
cursor.close()
conn.close()

import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine, event
from urllib.parse import quote_plus


conn =  "DRIVER={ODBC Driver 17 for SQL Server};SERVER=ecommerceta.database.windows.net;DATABASE=ecommerceta;UID=knight;PWD=Arief-1305"
quoted = quote_plus(conn)
new_con = 'mssql+pyodbc:///?odbc_connect={}'.format(quoted)
engine = create_engine(new_con)


@event.listens_for(engine, 'before_cursor_execute')
def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
    print("FUNC call")
    if executemany:
        cursor.fast_executemany = True


newtable = 'ecommercetrends_sepatupria'

s = time.time()
dataset.to_sql(newtable, engine, if_exists = 'append', index=False, chunksize=10000)
print(time.time() - s)