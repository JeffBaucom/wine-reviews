from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import numpy as np
import csv
#import sys
import os
import pandas

#reload(sys)
#sys.setdefaultencoding('utf8')

class WineDictionary(object):
    def __init__(self):
        self.fN = os.path.relpath('raw/raw_wine_data.csv')

    def get_dictionary(self):
        data = pandas.read_csv(self.fN, sep=',', encoding='utf-8')

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_result = vectorizer.fit_transform(data.description)
        scores = zip(vectorizer.get_feature_names(), np.asarray(tfidf_result.sum(axis=0)).ravel())
        return sorted(scores, key=lambda x: x[1], reverse=True)

class TaggedWineDocument(object):
    def __init__(self, tokes, titles):
        self.tokes = tokes
        self.titles = titles
    
    def __iter__(self):
        for idx, wine in enumerate(self.tokes):
            yield TaggedDocument(simple_preprocess(wine), [self.titles[idx]])
            

"""
fileName = 'winemag-data-130k-v2.csv'
wine_dictionary = WineDictionary(fileName);
myData = wine_dictionary.write_dictionary();
print myData.loc[1].description
"""

