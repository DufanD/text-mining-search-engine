import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

read_data = pd.read_csv('artikel.csv', nrows=10)
read_data = read_data.replace('\n', ' ', regex=True)

input_search = [
  'gereja di bekas negara'
]

list_document = list()
vectorizer = TfidfVectorizer()

for i in range(0, len(read_data)):
  output   = stemmer.stem(read_data.loc[i][0])
  X = vectorizer.fit_transform([output])
  Y = vectorizer.transform(input_search)
  list_document.append([read_data.loc[i][0][0:10], sum(Y.data)])

print(sorted(list_document, key = lambda x: x[1], reverse = True)[0:5])