import nltk
from nltk import FreqDist
import unicodecsv as csv
from numpy import *
import random
import pickle


def main():
    classifier = load_classifier()
    print classifier.predict(["Anoukh is very happy", "Yes Good"])

    output = open("./../../Dataset/Output/logan-output.csv", "wb")
    writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    with open("./../../Dataset/logan_imdb_text_processed.csv", "rb") as f:
        reader_tweets = csv.reader(f)
        for row in list(reader_tweets):
            output_array = []
            output_array.append(row[0])
            output_array.append(row[1])
            output_array.append(classifier.predict(row[0]))
            writer.writerow(output_array)


def load_classifier():
    f = open('FiveClassBNNBClassifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


if __name__ == '__main__':
    main()

