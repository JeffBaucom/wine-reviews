from wine_classifier import WineClassifier
from word_classifier import WordClassifier
from wine_dictionary import WineDictionary
from caveman_sommelier import Caveman
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import codecs
import string
import numpy as np
import math
import csv

fileName = 'winemag-data-130k-v2.csv'
#stoplist = 'for , are its wine it\'s a of the an it with is this that from but also while on and to in'
stopArr =  [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
stoplist = ' '.join(map(str, stopArr))

# Constructing fuzzy search
original_categories = {
        'fruit' : ['plum', 'currant', 'cherry', 'blueberry', 'pomegranate', 'cranberry', 'berry', 'apple', 'blackberry', 'citrus'],
        'spice' : ['smoke', 'cocoa', 'spiced', 'leathery', 'spicy', 'molasses', 'woodspice', 'cedar'],
        'floral' : ['floral', 'aromatic', 'perfume', 'rose', 'petal', 'hibiscus', 'geranium', 'lavender', 'jasmine', 'violet'],
        'oak' : ['wood', 'barrel', 'oaky', 'chocolaty', 'raisiny', 'syrupy', 'woody'],
        'herb' : ['mint', 'sage', 'leaf', 'tobacco', 'bramble', 'stalky', 'leafy', 'minty', 'medicinal'],
        'inorganic' : ['mineral', 'minerality', 'flinty', 'rubbery', 'tar', 'menthol', 'graphite'],
    }

class WineBoard(object):
    """
    Driver class for the project
    TODO: Pull processing and calculations into a separate class

    Attributes
    ----------
    added_terms : dictionary<str: [str]>
        list of terms added in most recent pass
    wordlist: [str]
        list of words already added to the training set
    x_train: [[float]]
        list of vectors that represents the training set raw values
    y_train: [[int]]
        matrix of integers that represent the categories of the corresponding x_train elements
    """

    def __init__(self, stopList, categories, keywords, weights):
        self.added_terms = {}
        self.wordlist = []
        self.x_train = []
        self.y_train = []
        self.x_wine_train = []
        self.y_wine_train = []
        self.x_wine_pred = []
        self.y_wine_pred = []

        tolerance = 300
        reviewCount = 100000
        dictionary = 'wine_dictionary.csv'
        caveman = 'caveman_data.csv'
        board = 'wine_board.csv'
        self.cutoff = 0.80
        self.dupeCutoff = .90
        self.misses = 0
        self.passes = 0

        myMan = Caveman(fileName, stoplist)
        tokes = myMan.write_reviews(caveman, reviewCount)
        self.locs = myMan.parse_geovariety(reviewCount)
        #print self.locs
        myDictionary = WineDictionary(fileName)
        self.vocab = myDictionary.write_dictionary(dictionary, tolerance)
        self.categories = categories
        self.added_categories = {}
        for i in self.categories:
            self.added_categories[i] = []
        self.added_terms = {}

        self.wordnet_lemmatizer = WordNetLemmatizer()
        fname = "WineTermModels"
        print self.added_categories

        #model = Word2Vec(tokes)
        try:
            print "Loaded"
            self.word_vectors = Word2Vec.load(fname)
        except:
            print "Not Loaded"
            self.word_vectors = Word2Vec(tokes)

        self.given_categories = {}
        self.original_categories = {} 
        for index in range(len(categories)):
            print index
            self.given_categories[categories[index]] = keywords[index]
            self.original_categories[categories[index]] = keywords[index]

        print self.given_categories

        for key, val in self.given_categories.iteritems():
            self.wordlist = self.wordlist + val

        self.WordClassifier = WordClassifier('distance', 7, self.categories)
        self.WineClassifier = WineClassifier('distance', 7, self.categories)

        self.cutoffs = {}
        for i in categories:
            self.cutoffs[i] = weights[self.categories.index(i)]


    def calculator_helper(self, term, category):
        lem = self.wordnet_lemmatizer.lemmatize(term)
        term_sum = 0
        term_ct = len(self.given_categories[category])
        catList = self.given_categories[category]
        maxVal = 0
        #if category not in catList:
        #    catList.append(category)
        for given_term in catList:
            given_lem = self.wordnet_lemmatizer.lemmatize(given_term)
            try:
                #try to find distance
                term_distance = self.word_vectors.similarity(term, given_term)
            except:
                try:
                    term_distance = self.word_vectors.similarity(lem, given_term)
                except:
                    try:
                        term_distance = self.word_vectors.similarity(term, given_lem)
                    except:
                        try:
                            term_distance = self.word_vectors.similarity(lem, given_lem)
                        except:
                            self.misses += 1
                            term_distance = 0
                            term_ct -= 1 
            if term_distance > maxVal and term_distance < 1:
                maxVal = term_distance
            term_sum += term_distance

        term_avg = 0
        if term_sum != 0:
            term_avg = term_sum/term_ct
            term_avg += (maxVal - term_avg)/2

        return term_avg

    def comb_categories(self):
        num = 5

        for key, value in self.vocab.iteritems():
            try:
                term_list = self.word_vectors.most_similar(self.wordnet_lemmatizer.lemmatize(key), topn=num)
                item_dict = {}
                item_dict['vocab'] = key
                item_dict['weight'] = math.floor(value)
                item_dict['max'] = term_list[num - 1][1]
                for catKey, catVal in self.given_categories.iteritems():
                    if key in self.given_categories[catKey] and item_dict['max'] > self.cutoffs[catKey]:
                        for i in term_list:
                            if i[0] not in self.added_categories[catKey] and i[0] not in self.given_categories[catKey] and i[0] not in self.wordlist and i[0][1] > self.cutoffs[catKey]:
                                self.added_categories[catKey].append(i[0])
                                self.wordlist.append(i[0])
                                if i[0] not in self.added_terms.keys():
                                    self.added_terms[i[0]] = set()
                                self.added_terms[i[0]].add(catKey)
                            elif i[0] not in self.added_categories[catKey] and i[0] not in self.given_categories[catKey] and i[0][1] > self.dupeCutoff:
                                self.added_categories[catKey].append(i[0])
                                if i[0] not in self.added_terms.keys():
                                    self.added_terms[i[0]] = set()
                                self.added_terms[i[0]].add(catKey)
                    elif term_list[0][0] in self.given_categories[catKey] and item_dict['max'] > self.cutoffs[catKey]:
                        if key not in self.added_categories[catKey] and key not in self.given_categories[catKey] and key not in self.wordlist:
                            self.added_categories[catKey].append(key)
                            self.wordlist.append(key)
                            if key not in self.added_terms.keys():
                                self.added_terms[key] = set()
                            self.added_terms[key].add(catKey)
                        elif key not in self.added_categories[catKey] and key not in self.given_categories[catKey] and item_dict['max'] > self.dupeCutoff:
                            self.added_categories[catKey].append(key)
                            if key not in self.added_terms.keys():
                                self.added_terms[key] = set()
                            self.added_terms[key].add(catKey)

            except:
                pass
                print "{} not found in text".format(key)

    def print_results(self):
        print "PASS {}".format(self.passes)
        print "--Added This Pass--"
        for key, val in self.added_categories.iteritems():
            print "{} : {}".format(key, val)
        print "--Total Added--"
        for key, val in self.added_categories.iteritems():
            outList = val + self.given_categories[key]
            self.given_categories[key] = outList
            print "{} : {}".format(key, outList)

    def calculate_distances(self):
        for key, val in self.original_categories.iteritems():
            for word in val:
                if word not in self.added_terms.keys():
                    self.added_terms[word] = set()
                self.added_terms[word].add(key)
        for key, val in self.added_terms.iteritems():
            # for each added list
            #for each added term
            vec = []
            for category in self.categories:
                vec.append(self.calculator_helper(key, category))

            self.x_train.append(vec)
            self.y_train.append([key, val])

    def reshuffle_results(self):
        for key, val in self.added_categories.iteritems():
            outList = val + self.given_categories[key]
            self.given_categories[key] = outList
            self.added_categories[key] = []

    def write_vectors(self):
        categories = self.categories
        categories.insert(0, "category")
        categories.insert(0, "term")
        csvRows = []
        csvRows.append(categories)
        for i in range(len(X)):
            csvRows.append(y[i] + X[i])
            print "{} -- {}".format(X[i], y[i])

        myFile = open('vectors.csv', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(csvRows)

    def train_words(self, n):
        #for i in range(n):
            #self.comb_categories()
            #self.print_results()
            #self.reshuffle_results()

        self.calculate_distances()
        self.all_words = np.array(self.y_train)
        self.all_words = self.all_words[:, 0]
        print self.all_words
        self.x_train, self.y_train = self.WordClassifier.transform_training(self.x_train, self.y_train)
        self.WordClassifier.train(self.x_train, self.y_train)

    def train_wines(self, n):
        words = self.parse_reviews(n)
        vector = self.calculate_reviews(n, words)
        categories_matrix = self.parse_categories(n)
        mid = n/2
        training_vector = vector[0:mid]
        prediction_vector = vector[mid:len(vector)]
        self.y_wine_train = categories_matrix[0:mid]
        self.y_wine_pred = categories_matrix[mid:len(words)]

        raw_words = []
        outs = []
        for i in range(len(vector)):
            wine_words = self.WordClassifier.transform_prediction(vector[i])
            raw_words.append(wine_words)
            outs.append(self.WordClassifier.predict(wine_words))

        outs = np.array(outs)
        np_vectors = np.array(vector)
        for j in range(len(outs)):
            outs[j] = np.mean(outs[j], axis=0)
            np_vectors[j] = np.mean(np_vectors[j], axis=0)
            outs[j] = np.append(outs[j], np_vectors[j])

        self.x_wine_train = outs[0:mid]
        self.x_wine_pred = outs[mid:len(outs)]

        self.x_wine_train, self.y_wine_train = self.WineClassifier.transform_training(self.x_wine_train, self.y_wine_train, self.locs)
        self.WineClassifier.train(self.x_wine_train, self.y_wine_train)
        self.x_wine_pred = self.WineClassifier.transform_prediction(self.x_wine_pred)
        result = self.WineClassifier.predict(self.x_wine_pred)
        resultMatrix = []
        for i in range(0, len(result)):
            wine = result[i]
            if i < len(self.y_wine_pred):
                expected = ' '.join(self.y_wine_pred[i])
            else:
                expected = "Not Trained"
            resultMatrix.append([])
            resultMatrix[i].append(expected)
            resultMatrix[i].append(set())
            resultMatrix[i].append(words[mid + i])
            for idx in range(0, len(wine)):
                if wine[idx] == 1:
                    resultMatrix[i][1].add(self.categories[idx])
#        for i in range(0, len(resultMatrix)):
#            if i < len(self.y_wine_pred):
#                print "Expected: {} - Result: {}".format(resultMatrix[i][0], list(resultMatrix[i][1]))
#            else:
#                print "Result: {}".format(list(resultMatrix[i][1]))
#                print ' '.join(words[i])

#            print "Review: \"{}\"".format(i[2])
        transformed_raw = []
        raw_words = np.array(raw_words)
        for i in raw_words:
            for j in i:
                transformed_raw.append(j)

        raw_words = np.concatenate((self.x_train, np.array(transformed_raw)), axis=0)
        def onpick(event):
            ind = event.ind
            print event
            print self.all_words[ind]
        fig, ax = plt.subplots()
        col = ax.scatter(raw_words[:, 0], raw_words[:, 1], c='r', alpha=0.5, picker=True)
        fig.canvas.mpl_connect('pick_event', onpick)
        print "Please show plot"

    def parse_reviews(self, n):
        reviews = []
        with codecs.open('caveman_data.csv', 'r', encoding='utf-8', errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile)
            sents = []
            index = 0

            while index < n:
                nextDict = reader.next()
                description = nextDict['review']
                sents.append(sent_tokenize(description))
                index += 1

            for reviewIdx, val in enumerate(sents):
                review = []
                for sentIdx, sent in enumerate(val):
                    for word in sent.lower().split():
                        review.append(word)

                reviews.append(review)

        return reviews

    def parse_categories(self, n):
        with codecs.open('training_wines_2.csv', 'r', encoding='utf-8', errors='ignore') as trainingfile:
            reader = csv.DictReader(trainingfile)
            out = []
            index = 0
            while index < n:
                try:
                    nextDict = reader.next()
                except:
                    break
                wine_categories = nextDict['categories']
                wine_categories = wine_categories.lower().translate(None, string.punctuation).split()

                out.append(wine_categories)
                index += 1

        return out


    def calculate_reviews(self, n, reviews):
        totalReviews = []

        for i in range(0, n): 
            reviewVector = []
            for word in reviews[i]:
                self.all_words = np.append(self.all_words, word)
                wordVector = []
                for category in self.categories:
                    result = self.calculator_helper(word, category)
                    wordVector.append(result)
                reviewVector.append(wordVector)
            totalReviews.append(reviewVector)
        return totalReviews

keywords = [
        ['blackberry', 'blackcherry', 'boysenberry', 'blueberry', 'blackberry', 'mulberry', 'cassis', 'plum', 'darkfruit', 'darkskinned', 'black'],
        ['strawberry', 'raspberry', 'pomegranate', 'cranberry', 'currant', 'cherry', 'red'],
        ['grapefruit', 'lemon', 'lime', 'zest', 'peel', 'rind', 'mandarin', 'orange', 'sour', 'citrus'],
        ['pineapple', 'mango', 'guava', 'lychee', 'banana', 'passion', 'melon', 'tropical', 'passion', 'kiwi'],
        ['pear', 'apple', 'peach', 'apricot', 'stonefruit', 'nectarine', 'orchard', 'yellow', 'yellowfruit'],
        ['wood', 'woody', 'toast', 'cream', 'creamy', 'coconut', 'oaky', 'coffee', 'butter', 'buttered', 'cigar', 'hickory', 'chocolate', 'caramel', 'nut', 'barrel'],
        ['spiced', 'spicy', 'cinnamon', 'nutmeg', 'clove', 'cardamom', 'anise', 'cocoa', 'pepper', 'licorice', 'peppercorn', 'baking', 'ginger'],
        ['honey', 'honeysuckle', 'lavender', 'jasmine', 'rose', 'violet', 'blossom', 'chamomile', 'flower', 'floral'],
        ['tomato', 'lettuce', 'tobacco', 'eucalyptus', 'hay', 'leafy', 'cilantro', 'pepper', 'green', 'ivy'],
        ['sage', 'thyme', 'mint', 'grass', 'medicinal', 'juniper', 'herb', 'herbaceous', 'rosemary', 'tarragon', 'green', 'marjoram'],
        ['menthol', 'forest', 'bramble', 'leather', 'musk', 'truffle', 'floor', 'balsamic', 'smoke', 'espresso', 'mineral', 'tar', 'flinty', 'minerality', 'graphite', 'rubbery', 'gritty', 'rugged', 'earthy', 'sidewalk']]
categories = ['black', 'red', 'citrus', 'tropical', 'tree', 'oak', 'spice', 'floral', 'vegetal', 'herb', 'earthy']
weights =    [0.72,    0.65,  0.75,     0.77,       0.69,   0.70,   0.76,    0.77,     0.67,      0.65,   0.83]
given_categories = original_categories

#print categories
myBoard = WineBoard(stoplist, categories, keywords, weights)
#myBoard.train_words(4)
#myBoard.train_wines(220)
#plt.show()
