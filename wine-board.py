from wine_dictionary import WineDictionary
from gensim.models import Word2Vec
from caveman_sommelier import Caveman
from nltk.stem import WordNetLemmatizer
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.decomposition import PCA
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
        'fruit' : ['plum', 'currant', 'juicy', 'jelly', 'cherry', 'blueberry', 'pomegranate', 'cranberry', 'berry', 'apple', 'blackberry', 'citrus'],
        'spice' : ['smoke', 'cocoa', 'spiced', 'leathery', 'spicy', 'baked', 'roasted', 'molasses', 'woodspice', 'cedar'],
        'floral' : ['floral', 'aromatic', 'delicate', 'fragrant', 'perfume', 'rose', 'petal', 'hibiscus', 'geranium', 'lavender', 'jasmine', 'violet'],
        'oak' : ['wood', 'barrel', 'oaky', 'chocolaty', 'raisiny', 'syrupy', 'woody'],
        'herb' : ['mint', 'sage', 'leaf', 'tobacco', 'bramble', 'stalky', 'leafy', 'minty', 'saucy', 'medicinal'],
        'inorganic' : ['mineral', 'minerality', 'flinty', 'lend', 'rubbery', 'tar', 'menthol', 'graphite'],
    }
#{
#        'fruit': ['jammy', 'ripe', 'juicy', 'fleshy', 'plummy', 'berry', 'cassis', 'citrus', 'stonefruit', 'tropicalfruit', 'redfruit', 'melon', 'apple', 'pear', 'mango', 'lime', 'cherry'],
#        'spice': ['pepper', 'clove', 'anise', 'cinammon', 'nutmeg', 'saffron', 'ginger', 'spicy'], 
#        'floral': ['hibiscus', 'potpourri', 'rose', 'lavender',  'geranium', 'blossom', 'violet', 'jasmine'],
#        'oak': ['smoke', 'smoky', 'vanilla', 'cocoa', 'cream', 'coffee', 'butter'],
#        'herb': ['vegetal', 'vegetable', 'asparagus', 'grass', 'sage', 'eucalyptus', 'dill', 'quince', 'green'],
#        'inorganic': ['mineral', 'graphite', 'petroleum', 'plastic', 'rubber', 'tar']
#        }

given_categories = original_categories
categories = ['fruit', 'spice', 'floral', 'oak', 'herb', 'inorganic']
added_categories = {'fruit': [], 'spice': [], 'floral': [], 'inorganic': [], 'herb': [], 'oak': []}

wordlist = []
for key, val in given_categories.iteritems():
    wordlist = wordlist + val



tolerance = 300
reviewCount = 30000
dictionary = 'wine_dictionary.csv'
caveman = 'caveman_data.csv'
board = 'wine_board.csv'
num = 5
cutoff = 0.70
dupeCutoff = .90
misses = 0

myMan = Caveman(fileName, stoplist)
tokes = myMan.write_reviews(caveman, reviewCount)
myDictionary = WineDictionary(fileName)
vocab = myDictionary.write_dictionary(dictionary, tolerance)

wordnet_lemmatizer = WordNetLemmatizer()

model = Word2Vec(tokes)
word_vectors = Word2Vec(tokes)
csvList = []
passes = 0

#initialize vector lists
X = []
y = []

def comb_categories(passes):
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
                        if i[0] not in added_categories[catKey] and i[0] not in given_categories[catKey] and i[0] not in wordlist and i[0][1] > cutoff:
                            added_categories[catKey].append(i[0])
                            wordlist.append(i[0])
                        elif i[0] not in added_categories[catKey] and i[0] not in given_categories[catKey] and i[0][1] > dupeCutoff:
                            added_categories[catKey].append(i[0])
                elif term_list[0][0] in given_categories[catKey] and item_dict['max'] > cutoff:
                    if key not in added_categories[catKey] and key not in given_categories[catKey] and key not in wordlist:
                        added_categories[catKey].append(key)
                        wordlist.append(key)
                    elif key not in added_categories[catKey] and key not in given_categories[catKey] and item_dict['max'] > dupeCutoff:
                        added_categories[catKey].append(key)

            for idx, val in enumerate(term_list):
                string = "word" + str(idx + 1)
                item_dict[string] = val[0]
            csvList.append(item_dict)

        except:
            pass
            #print "{} not found in text".format(key)

    passes += 1

def print_results(passes):
    print "PASS {}".format(passes)
    print "--Added This Pass--"
    for key, val in added_categories.iteritems():
        print "{} : {}".format(key, val)
    print "--Total Added--"
    for key, val in added_categories.iteritems():
        outList = val + given_categories[key]
        given_categories[key] = outList
        print "{} : {}".format(key, outList)

def calculate_distances():
    for key, val in added_categories.iteritems():
        # for each added list
        for term in val:
            #for each added term
            vec = []
            for category in categories:
                vec.append(calculator_helper(term, category))
                

            X.append(vec)
            y.append([term, key])

def calculator_helper(term, category):
    global misses
    lem = wordnet_lemmatizer.lemmatize(term)
    term_sum = 0
    term_ct = len(given_categories[category])
    catList = given_categories[category]
    maxVal = 0
    if category not in catList:
        catList.append(category)
    for given_term in catList:
        given_lem = wordnet_lemmatizer.lemmatize(given_term)
        try:
            #try to find distance
            term_distance = word_vectors.similarity(term, given_term)
        except:
            try:
                term_distance = word_vectors.similarity(lem, given_term)
            except:
                try:
                    term_distance = word_vectors.similarity(term, given_lem)
                except:
                    try:
                        term_distance = word_vectors.similarity(term, given_lem)
                    except:
                        try:
                            term_distance = word_vectors.similarity(lem, given_lem)
                        except:
                            misses += 1
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

    return maxVal



def reshuffle_results():
    for key, val in added_categories.iteritems():
        outList = val + given_categories[key]
        given_categories[key] = outList
        added_categories[key] = []

def run_neighbors():


    n_neighbors = 11

    h = .02
    global X
    global y
    print(X)

    pca = PCA(n_components = 2)
    X2D = pca.fit_transform(X)
    print(X2D)

    X = np.array(X2D)
    y = np.array(y)
    print X
    y = y[:, 1:2]
    y = y.flatten()
    nY = []
    for val in np.nditer(y):
        if val == u'fruit':
            nY.append(0)
        elif val == u'floral':
            nY.append(1)
        elif val == u'inorganic':
            nY.append(2)
        elif val == u'herb':
            nY.append(3)
        elif val == u'oak': #  or 
            nY.append(4)
        else:
            nY.append(5)

    y = np.array(nY)

    print y

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#800080', '#D2691E'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#8B4513'])

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))

    plt.show()

def write_vectors():
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

def comb_and_calculate(n):
    global passes
    for i in range(n):
        comb_categories(passes)
        print_results(passes)
        calculate_distances()
        reshuffle_results()

comb_and_calculate(3)
print misses
write_vectors()
run_neighbors()




#sortedCsv = sorted(csvList, key=lambda k: k['weight'], reverse=True)
#with open(board, 'wb') as csvfile:
#    fieldnames = ['vocab', 'weight']
#    for i in range(1, num + 1):
#        fieldnames.append('word' + str(i))
#    fieldnames.append('max')
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    writer.writeheader()
#    writer.writerows(sortedCsv)

