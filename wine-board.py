from wine_dictionary import WineDictionary
from gensim.models import Word2Vec
from caveman_sommelier import Caveman
from nltk.stem import WordNetLemmatizer
import math
import csv

fileName = 'winemag-data-130k-v2.csv'
stoplist = 'for , are its wine it\'s a of the an it with is this that from but also while on and to in'
tolerance = 300
reviewCount = 30000
dictionary = 'wine_dictionary.csv'
caveman = 'caveman_data.csv'
board = 'wine_board.csv'
num = 5

myMan = Caveman(fileName, stoplist)
tokes = myMan.write_reviews(caveman, reviewCount)
myDictionary = WineDictionary(fileName)
vocab = myDictionary.write_dictionary(dictionary, tolerance)

wordnet_lemmatizer = WordNetLemmatizer()

model = Word2Vec(tokes)
word_vectors = Word2Vec(tokes)
csvList = []
for key, value in vocab.iteritems():
    try:
        term_list = word_vectors.most_similar(wordnet_lemmatizer.lemmatize(key), topn=num)
        item_dict = {}
        item_dict['vocab'] = key
        item_dict['weight'] = math.floor(value)
        item_dict['max'] = term_list[num - 1][1]

        for idx, val in enumerate(term_list):
            string = "word" + str(idx + 1)
            item_dict[string] = val[0]
        csvList.append(item_dict)

    except:
        pass
        #print "{} not found in text".format(key)

sortedCsv = sorted(csvList, key=lambda k: k['weight'], reverse=True)
with open(board, 'wb') as csvfile:
    fieldnames = ['vocab', 'weight']
    for i in range(1, num + 1):
        fieldnames.append('word' + str(i))
    fieldnames.append('max')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sortedCsv)

