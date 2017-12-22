from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import pandas
import sys
import string
import numpy as np
import csv

reload(sys)
sys.setdefaultencoding('utf8')
stoplist = set('for , are its wine it\'s a of the an it with is this that from but also while on and to in'.split())

wordnet_lemmatizer = WordNetLemmatizer()


fileName = 'winemag-data-130k-v2.csv'
def read_toke(fileName):
    with open(fileName, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        sents = []
        index = 0
        #Change ^      v   these numbers to read different reviews
        while index < 30000:
            nextDict = reader.next()
            description = nextDict['description']
            sents.append(sent_tokenize(description))
            index += 1;

        tokes = [] 
        for idx, val in enumerate(sents):
            # for each description
            for idx, sent in enumerate(val):
            # for each sentence in each description
                out = []
                for word in sent.lower().split():
                    if word not in stoplist:
                        no_punc_word = word.translate(None, string.punctuation)
                        if len(no_punc_word):
                            out.append(unicode(wordnet_lemmatizer.lemmatize(no_punc_word)))
                tokes.append(out)
        print tokes
        return tokes


def read_tfidf(fileName):
    #count_vect = CountVectorizer()
    data = pandas.read_csv(fileName, sep=',', encoding='utf-8')
    #X_train_counts = count_vect.fit_transform(data.description)
    #print X_train_counts.shape

    #tfidf_transformer = TfidfTransformer()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_result = vectorizer.fit_transform(data.description)
    #X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #print X_train_tfidf.shape
    scores = zip(vectorizer.get_feature_names(), np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=False)
    out = {}
    for item in sorted_scores:
        if item[1] > 400:
            #print "{0:50} Score: {1}".format(item[0], item[1])
            out[item[0]] = item[1]
    return out

sent_tokes = read_toke(fileName)
sort_scores = read_tfidf(fileName)
#print sort_scores
model = Word2Vec(sent_tokes)
word_vectors = model.wv
for key, value in sort_scores.iteritems():
    try:
        print "{}: {}".format(key, word_vectors.most_similar(wordnet_lemmatizer.lemmatize(key), topn=5))
    except:
        print "{} not found in text".format(key)

