from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import csv
import sys
import string
import codecs

reload(sys)
sys.setdefaultencoding('utf8')

class Caveman(object):
    # takes fileName of CSV to read
    # takes stoplist as a string of space separated stop words
    def __init__(self, fileName, stoplist):
        self.fN = fileName
        self.stoplist = set(stoplist.split())

    def write_reviews(self, outName, reviewCount):
        with codecs.open(self.fN, 'r', encoding='utf-8', errors='ignore') as csvfile:
            wordnet_lemmatizer = WordNetLemmatizer()
            reader = csv.DictReader(csvfile)
            sents = []
            index = 0
            while index < reviewCount:
                nextDict = reader.next()
                description = nextDict['description']
                sents.append(sent_tokenize(description))
                index += 1

            tokes = []
            reviews = []
            reviews.append(['review'])
            #iterate over each row(each review)
            for idx, val in enumerate(sents):
                review = ""
                #iterate over each sentence
                for idx, sent in enumerate(val):
                    sentence = []
                    #iterate over each word
                    for word in sent.lower().split():
                        if word not in self.stoplist:
                            no_punc_word = word.translate(None, string.punctuation)
                            if len(no_punc_word):
                                sentence.append(unicode(wordnet_lemmatizer.lemmatize(no_punc_word)))
                                review += wordnet_lemmatizer.lemmatize(no_punc_word) + " "
                    tokes.append(sentence)
                reviews.append([review])
            with open(outName, 'wb') as outFile:
                fieldnames = ['review']
                writer = csv.writer(outFile)
                writer.writerows(reviews)
            return tokes
