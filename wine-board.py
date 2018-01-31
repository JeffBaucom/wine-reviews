from wine_dictionary import WineDictionary
from gensim.models import Word2Vec
from caveman_sommelier import Caveman
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import codecs
import matplotlib.pyplot as plt
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
keywords = [['blackberry', 'blackcherry', 'boysenberry', 'blueberry', 'blackberry'], ['strawberry', 'raspberry', 'pomegranate', 'cranberry', 'currant', 'cherry'], ['grapefruit', 'lemon', 'lime', 'zest', 'peel', 'rind', 'mandarin', 'orange'], ['pineapple', 'mango', 'guava', 'lychee', 'banana', 'passion', 'melon'], ['pear', 'apple', 'peach', 'apricot', 'stonefruit', 'honey'], ['wood', 'woody', 'toast', 'cream', 'creamy', 'coconut', 'oaky', 'coffee', 'butter', 'buttered'], ['spiced', 'spicy', 'cinnamon', 'nutmeg', 'clove', 'cardamom', 'anise', 'cocoa', 'pepper', 'licorice', 'peppercorn'], ['honeysuckle', 'lavender', 'jasmine', 'rose', 'violet', 'blossom', 'chamomile'], ['tomato', 'lettuce', 'tobacco', 'eucalyptus', 'hay', 'leafy'], ['sage', 'thyme', 'mint', 'grass', 'medicinal', 'juniper'], ['menthol', 'forest', 'bramble', 'leather', 'musk', 'truffle', 'floor', 'balsamic', 'smoke', 'espresso', 'mineral', 'tar', 'flinty', 'minerality', 'graphite', 'rubbery', 'gritty', 'rugged']]
#{
#        'fruit': ['jammy', 'ripe', 'juicy', 'fleshy', 'plummy', 'berry', 'cassis', 'citrus', 'stonefruit', 'tropicalfruit', 'redfruit', 'melon', 'apple', 'pear', 'mango', 'lime', 'cherry'],
#        'spice': ['pepper', 'clove', 'anise', 'cinammon', 'nutmeg', 'saffron', 'ginger', 'spicy'], 
#        'floral': ['hibiscus', 'potpourri', 'rose', 'lavender',  'geranium', 'blossom', 'violet', 'jasmine'],
#        'oak': ['smoke', 'smoky', 'vanilla', 'cocoa', 'cream', 'coffee', 'butter'],
#        'herb': ['vegetal', 'vegetable', 'asparagus', 'grass', 'sage', 'eucalyptus', 'dill', 'quince', 'green'],
#        'inorganic': ['mineral', 'graphite', 'petroleum', 'plastic', 'rubber', 'tar']
#        }

categories = ['black', 'red', 'citrus', 'tropical', 'tree', 'oak', 'spice', 'floral', 'vegetal', 'herb', 'earthy']
given_categories = original_categories


#def run_neighbors(test):
    #y = y.flatten()
#    nY = []
#    for val in np.nditer(y):
#        if val == u'fruit':
#            nY.append(0)
#        elif val == u'floral':
#            nY.append(1)
#        elif val == u'inorganic':
#            nY.append(2)
#        elif val == u'herb':
#            nY.append(3)
#        elif val == u'oak': #  or 
#            nY.append(4)
#        else:
#            nY.append(5)
#
#    y = np.array(nY)


    # Create color maps
    #cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#800080', '#D2691E'])
    #cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#8B4513'])

    #for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        #clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        #print clf.fit(X, y)
        #print clf.predict(t)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        #x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        #y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        #xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
        #                     np.arange(y_min, y_max, h))
        #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        #Z = Z.reshape(xx.shape)
        #plt.figure()
        #plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        #plt.scatter(X[:, 0], X[:, 1], c='gray', edgecolor=(0, 0 , 0), s=260)
        #zero_class = np.where(y[:, 0])
        #first_class = np.where(y[:, 1])
        #second_class = np.where(y[:, 2])
        #third_class = np.where(y[:, 3])
        #fourth_class = np.where(y[:, 4])
        #fifth_class = np.where(y[:, 5])
        #plt.scatter(X[zero_class, 0], X[zero_class, 1], c='red', edgecolor='k', s=100, facecolors='none')
        #plt.scatter(X[first_class, 0], X[first_class, 1], c='green', edgecolor='k', s=80, facecolors='none')
        #plt.scatter(X[second_class, 0], X[second_class, 1], c='blue', edgecolor='k', s=40, facecolors='none')
        #plt.scatter(X[third_class, 0], X[third_class, 1], c='orange', edgecolor='k', s=20, facecolors='none')
        #plt.scatter(X[fourth_class, 0], X[fourth_class, 1], c='yellow', edgecolor='k', s=10, facecolors='none')
        #plt.scatter(X[fifth_class, 0], X[fifth_class, 1], c='black', edgecolor='k', s=5, facecolors='none')
        #plt.xlim(xx.min(), xx.max())
        #plt.ylim(yy.min(), yy.max())
        #plt.title("3-Class classification (k = %i, weights = '%s')"
                  #% (n_neighbors, weights))

    #plt.show()




#comb_and_calculate(3)
#print misses
#write_vectors()
#test = calculate_reviews(10)
#word_vectors.save(fname)
#run_neighbors(test)

class WordClassifier(object):

    def __init__(self, weights, n_neighbors, categories):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        self.categories = categories
        print categories
        self.pca = PCA(n_components = 3) 

    def train(self, x_train, y_train):
        h = .02

        X2D = self.pca.fit_transform(x_train)

        nY = []
        for i in y_train:
            tempList = list(i[1])
            for j in range(len(tempList)):
                val = tempList[j]
                tempList[j] = self.categories.index(tempList[j])
            nY.append(sorted(tempList))

        X = np.array(X2D)
        y = MultiLabelBinarizer().fit_transform(nY)
        return self.clf.fit(X, y)

    def test(self, x_test, y_test):
        X2D = self.pca.transform(x_test)
        print y_test
        y_predict = self.clf.predict(X2D)
        print y_test
        print y_predict
        #print accuracy_score(y_test, y_predict)

    def predict(self, x_test):
        X2D = self.pca.transform(x_test)
        return self.clf.predict(X2D)

class WineClassifier(object):

    def __init__(self, weights, n_neighbors, categories):
        self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        self.categories = categories
        print categories
        self.pca = PCA(n_components = 3) 

    def train(self, x_train, y_train):
        h = .02

        X2D = self.pca.fit_transform(x_train)

        nY = []
        for i in y_train:
            tempList = list(i[1])
            for j in range(len(tempList)):
                val = tempList[j]
                tempList[j] = self.categories.index(tempList[j])
            nY.append(sorted(tempList))

        X = np.array(X2D)
        y = MultiLabelBinarizer().fit_transform(nY)
        return self.clf.fit(X, y)

    def predict(self, x_test):
        X2D = self.pca.transform(x_test)
        return self.clf.predict(X2D)


class WineBoard(object):

    def __init__(self, stopList, categories, keywords):
        self.added_terms = {}
        self.wordlist = []
        self.x_train = []
        self.y_train = []

        tolerance = 300
        reviewCount = 50000
        dictionary = 'wine_dictionary.csv'
        caveman = 'caveman_data.csv'
        board = 'wine_board.csv'
        self.cutoff = 0.70
        self.dupeCutoff = .90
        self.misses = 0
        self.passes = 0

        myMan = Caveman(fileName, stoplist)
        tokes = myMan.write_reviews(caveman, reviewCount)
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
        for index in range(len(categories)):
            self.given_categories[categories[index]] = keywords[index]

        print self.given_categories

        for key, val in self.given_categories.iteritems():
            self.wordlist = self.wordlist + val

        self.WordClassifier = WordClassifier('distance', 7, self.categories)


    def calculator_helper(self, term, category):
        lem = self.wordnet_lemmatizer.lemmatize(term)
        term_sum = 0
        term_ct = len(self.given_categories[category])
        catList = self.given_categories[category]
        maxVal = 0
        if category not in catList:
            catList.append(category)
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

        # append averages
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
                    if key in self.given_categories[catKey] and item_dict['max'] > self.cutoff:
                        for i in term_list:
                            if i[0] not in self.added_categories[catKey] and i[0] not in self.given_categories[catKey] and i[0] not in self.wordlist and i[0][1] > self.cutoff:
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
                    elif term_list[0][0] in self.given_categories[catKey] and item_dict['max'] > self.cutoff:
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

        #self.passes += 1

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
        for i in range(n):
            self.comb_categories()
            self.print_results()
            self.reshuffle_results()

        self.calculate_distances()
        self.WordClassifier.train(self.x_train, self.y_train)
        #words = []
        #for i in ['brimstone', 'pineapple']:
        #    word = []
        #    for j in self.categories:
        #        word.append(self.calculator_helper(i, j))
        #    words.append(word)
        #print words
        #self.WordClassifier.test(words, [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]])

    #def predict_wines(self, n):
        #wines = self.calculate_reviews(n)

    def train_wines(self, n):
        words = self.parse_reviews(n)
        vector = self.calculate_reviews(n, words)

        outs = []
        for i in range(n):
            outs.append(self.WordClassifier.predict(vector[i]))

        outs = np.array(outs)
        for j in range(len(outs)):
            #print outs[j]
            for k in range(len(outs[j])):
                outs[j][k] = np.multiply(outs[j][k], len(words[j][k]))
            outs[j] = np.mean(outs[j], axis=0)
            #print outs[j]
            #print words[j]
            for val in range(len(self.categories)):
                print "{}: {}".format(self.categories[val], outs[j][val])
            print ' '.join(map(str, words[j]))
        


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

    def calculate_reviews(self, n, reviews):
        totalReviews = []

        for i in range(0, n): 
            reviewVector = []
            for word in reviews[i]:
                wordVector = []
                for category in self.categories:
                    result = self.calculator_helper(word, category)
                    wordVector.append(result)
                reviewVector.append(wordVector)
            totalReviews.append(reviewVector)
        return totalReviews


print categories
myBoard = WineBoard(stoplist, categories, keywords)
myBoard.train_words(4)
myBoard.train_wines(10)
