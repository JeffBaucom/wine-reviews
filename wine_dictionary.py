from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv
import sys
import pandas

reload(sys)
sys.setdefaultencoding('utf8')

class WineDictionary(object):
    def __init__(self, fileName):
        self.fN = fileName

    def write_dictionary(self, outName, tolerance):
        data = pandas.read_csv(self.fN, sep=',', encoding='utf-8')

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_result = vectorizer.fit_transform(data.description)
        scores = zip(vectorizer.get_feature_names(), np.asarray(tfidf_result.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        out = []
        ret = {}

        for item in sorted_scores:
            if item[1] > tolerance:
                itemDict = {}
                itemDict['word'] = unicode(item[0])
                itemDict['score'] = item[1]
                ret[unicode(item[0])] = item[1]
                out.append(itemDict)

        with open(outName, 'wb') as csvfile:
            fieldnames = ['word', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out)

        return ret


