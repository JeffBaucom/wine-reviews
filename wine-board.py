from wine_dictionary import WineDictionary
from gensim.models import Word2Vec
from caveman_sommelier import Caveman
from nltk.stem import WordNetLemmatizer
import math
import csv

fileName = 'winemag-data-130k-v2.csv'
#stoplist = 'for , are its wine it\'s a of the an it with is this that from but also while on and to in'
stopArr =  [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
stoplist = ' '.join(map(str, stopArr))

# Constructing fuzzy search
given_categories = {'fruit': ['lime', 'blueberry', 'lemon', 'orange', 'apple', 'pear', 'nectarine', 'peach', 'mango', 'pineapple'], 'spice': ['thyme', 'mint', 'anise', 'pepper'], 'floral': ['hibiscus', 'potpourri', 'rose', 'lavender', 'violet', 'jasmine'], 'aging': ['smoke', 'coconut', 'vanilla', 'cocoa', 'tobacco', 'coffee']}
added_categories = {'fruit': [], 'spice': [], 'floral': [], 'aging': []}

tolerance = 300
reviewCount = 30000
dictionary = 'wine_dictionary.csv'
caveman = 'caveman_data.csv'
board = 'wine_board.csv'
num = 5
cutoff = 0.5

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
        for catKey, catVal in given_categories.iteritems():
            if key in given_categories[catKey] and item_dict['max'] > cutoff:
                for i in term_list:
                    if i[0] not in added_categories[catKey] and i[0] not in given_categories[catKey] and i[0][1] > cutoff:
                        added_categories[catKey].append(i[0])
            elif term_list[0][0] in given_categories[catKey] and item_dict['max'] > cutoff:
                if key not in added_categories[catKey] and key not in given_categories[catKey]:
                    added_categories[catKey].append(key)

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
print added_categories

