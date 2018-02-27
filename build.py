import luigi
import csv
from src import wine_dictionary

class TfidfTransform(luigi.Task):

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

