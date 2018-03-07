from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import pandas as pd
import string
import os
import csv
import codecs

class Caveman(object):
    # takes fileName of CSV to read
    # takes stoplist as a string of space separated stop words
    def __init__(self):
        stopArr =  [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
        stoplist = ' '.join(map(str, stopArr))

        self.fN = os.path.relpath('raw/raw_wine_data.csv')
        self.stoplist = set(stoplist.split())

    def parse_geovariety(self, reviewCount):
        with codecs.open(self.fN, 'r', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile);
            index = 0
            varieties = set()
            objs = []
            while index < reviewCount:
                wine_obj = {}
                nextDict = reader.next()
                wine_obj['variety'] = nextDict['variety']
                varieties.add(nextDict['variety'])
                wine_obj['province'] = nextDict['province']
                wine_obj['country'] = nextDict['country']
                objs.append(wine_obj)
                index += 1

        return (objs, varieties)

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

    def tokenize_reviews(self):
        data = pd.read_csv(self.fN, sep=',', encoding='utf-8')

        data['description_tokes'] = data['description'].apply(self.toke_lemmatize)
        #data['description_tokes'] = data['description']
        #header = ['description', 'description_tokes']
        #data.to_csv('to_csv.csv', columns=header)
        return data

    def toke_lemmatize(self, text):
        lemmatizer = WordNetLemmatizer
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        review = sent_tokenize(text.lower())
        out = []
        for sent in review:
            new_sent = []
            for word in pos_tag(word_tokenize(sent)):
                if word[0] not in self.stoplist:
                    new_sent.append(lemmatizer.lemmatize(word[1], word[0].lower()))
            if len(new_sent) > 0:
                out.append(new_sent)
        return out

        
