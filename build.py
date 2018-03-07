import luigi
import csv
import pandas as pd
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
        return luigi.LocalTarget("data/caveman_reviews.csv")

    def run(self):
        caveman = caveman_sommelier.Caveman()
        tokes_df = caveman.tokenize_reviews()
        header = ['description', 'description_tokes']
        tokes_df.to_csv(self.output(), columns = header)
         
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
        pass

    def requires(self):
        return None

class VectorizeWords(luigi.Task):
    """
    """
    def output(self):
        return luigi.localTarget("data/word_category_vectors.csv")

    def run(self):
        pass

    def requires(self):
        pass
