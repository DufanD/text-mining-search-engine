from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

corpus = [
    ['This is the first document. This document is the second document. And this is the third one. Is this the first document?'],
    ['Sun bright. Dark sky']
]
input_search = [
  'Third document third',
  'First document'
]

list_document = list()
vectorizer = TfidfVectorizer()
for corpus_item in corpus:
  X = vectorizer.fit_transform(corpus_item)
  Y = vectorizer.transform(input_search)
  print(vectorizer.get_feature_names())
  list_document.append([corpus_item, sum(Y.data)])
  print(Y)

# print(list_document)
print(sorted(list_document, key = lambda x: x[1], reverse = True))
# print(X)
# print(Y.)