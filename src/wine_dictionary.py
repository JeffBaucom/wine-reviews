from sklearn.feature_extraction.text import TfidfVectorizer
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


"""
fileName = 'winemag-data-130k-v2.csv'
wine_dictionary = WineDictionary(fileName);
myData = wine_dictionary.write_dictionary();
print myData.loc[1].description
"""

