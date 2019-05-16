from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    ['This is the first document. This document is the second document. And this is the third one. Is this the first document?'],
    ['Sun bright. Dark sky']
]

input_search = [
  'Third document'
]

list_document = list()
vectorizer = CountVectorizer()
for corpus_item in corpus:
  X = vectorizer.fit_transform(corpus_item)
  Y = vectorizer.transform(input_search)
  list_document.append([corpus_item, sum(Y.data)])
  print(vectorizer.get_feature_names())
  print(Y)

print(sorted(list_document, key = lambda x: x[1], reverse = True))