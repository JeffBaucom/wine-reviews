import luigi
import csv
import pandas as pd
from gensim.models import word2vec, doc2vec
from src import wine_dictionary, caveman_sommelier

class TfidfTransform(luigi.Task):
    """
    Reads the raw data and outputs the sorted words
    by TFIDF weight to a csv
    Parameters: with or without lemmatization
    """

    def output(self):
        return luigi.LocalTarget("data/tfidf_transform.csv")

    def run(self):
        dictionary = wine_dictionary.WineDictionary()
        sorted_scores = dictionary.get_dictionary()
        with self.output().open('w') as csvfile:
            fieldnames = ['word', 'score']
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow(fieldnames)
            for line in sorted_scores:
                writer.writerow([line[0], line[1]])

class WriteCaveman(luigi.Task):
    """
    Reads the raw data
    outputs a csv with lowercased and stop list removed reviews
    Parameters: with or without lemmatization
    """
    def output(self):
        return luigi.LocalTarget("data/caveman_pickle")

    def run(self):
        caveman = caveman_sommelier.Caveman()
        caveman.tokenize_reviews()
        tokes_df = caveman.tokenize_reviews()
        #header = ['description', 'description_tokes']
        f = self.output().path
        #tokes_df.to_csv(f)
        tokes_df.to_pickle(f)
        #f.close()
         
    def requires(self):
        return None

class TrainWord2Vec(luigi.Task):
    """
    Reads the raw data, and outputs a gensim model for the corpus
    using gensim.models.word2vec 
    Parameters: DBOW/CBOW
    """

    def output(self):
        return luigi.LocalTarget("data/models/gensim_word2vec")

    def run(self):
        data = pd.read_pickle(self.input().path)
        sents = data['description_tokes'].tolist()
        tokes = []
        for i in sents:
            tokes = tokes + i
        model = word2vec.Word2Vec(tokes, min_count=1, size=100, workers=4)
        model.save(self.output().path)

    def requires(self):
        return WriteCaveman()

class TrainDoc2Vec(luigi.Task):
    """
    Reads the raw data, and outputs a gensim model for the corpus
    using gensim.models.word2vec 
    Parameters: DBOW/CBOW, min_count, vector_size
    """

    def output(self):
        return luigi.LocalTarget("data/models/gensim_doc2vec")

    def run(self):
        data = pd.read_pickle(self.input().path)
        raw = pd.read_csv('raw/raw_wine_data.csv', sep=',', encoding='utf-8')
        titles = raw['title'].tolist()
        docs = data['description'].tolist() #pass this list to TaggedWine class
        documents = wine_dictionary.TaggedWineDocument(docs, titles)
        model = doc2vec.Doc2Vec(vector_size=50, window=8, min_count=2, workers=4)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(self.output().path)

    def requires(self):
        return WriteCaveman()

class VectorizeWords(luigi.Task):
    """
    """
    def output(self):
        return luigi.localTarget("data/word_category_vectors.csv")

    def run(self):
        pass

    def requires(self):
        pass
