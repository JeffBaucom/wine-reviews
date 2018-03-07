import luigi
import csv
from src import wine_dictionary

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

class TrainGensim(luigi.Task):
    """
    Reads the raw data, and outputs a gensim model for the corpus
    """

    def output(self):
        return luigi.LocalTarget("data/models/gensim")

    def run(self):
        dictionary = wine_dictionary.WineDictionary()
         


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
