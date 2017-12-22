from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import sys
import csv

reload(sys)
sys.setdefaultencoding('utf8')

idx = ''
desc = 'description'

fileName = 'winemag-data_first150k.csv'
with open(fileName) as csvfile:
    reader = csv.DictReader(csvfile)
    sents = []
    index = 0
    while index < 500:
        print index
        
        nextDict = reader.next()
        description = nextDict[desc]
        placeString = nextDict['country'] + ' - ' + nextDict['province'] + ' - ' + nextDict['region_1']  + ' - ' + nextDict['region_2'] + ' - ' + nextDict['winery']  
        print placeString

        sents.append(sent_tokenize(description))
        index += 1;


    tokes = [] 
    for idx, val in enumerate(sents):
        # for each description
        for idx, sent in enumerate(val):
        # for each sentence in each description
            tokes.append(word_tokenize(unicode(sent, 'utf-8')))
            
    vec = Word2Vec(tokes, size=100, window=2, min_count=5, workers=4, negative=20)
    print vec.most_similar('fruit', topn=10)
    print vec.most_similar('jammy', topn=10)
    print vec.most_similar('ripe', topn=10)
    print vec.most_similar('juicy', topn=10)
    print vec.most_similar('citrus', topn=10)
