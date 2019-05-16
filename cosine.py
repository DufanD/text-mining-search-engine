import pandas as pd
import pprint
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
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
vectorizer = CountVectorizer()

for i in range(0, len(read_data)):
  output = stemmer.stem(read_data.loc[i][0])
  X = vectorizer.fit_transform([output])
  Y = vectorizer.transform(input_search)
  cosine = cosine_similarity(X, Y)
  list_document.append([read_data.loc[i][0][0:25], cosine[0][0]])

pprint.pprint(sorted(list_document, key = lambda x: x[1], reverse = True)[0:5])