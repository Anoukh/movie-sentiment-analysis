import nltk
from nltk import FreqDist
import unicodecsv as csv
from numpy import *
import random
import pickle


def main():
    classifier = load_classifier()
    print classifier.classify(extract_features("Anoukh is very happy"))

    output = open("./../../Dataset/Output/logan-output.csv", "wb")
    writer = csv.writer(output, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    output_array = []
    with open("./../../Dataset/logan_imdb_text_processed.csv", "rb") as f:
        reader_tweets = csv.reader(f)
        for row in list(reader_tweets):
            output_array.append(row[0])
            output_array.append(row[1])
            output_array.append(classifier.classify(extract_features(row[0])))
            writer.writerow(output_array)


def load_classifier():
    f = open('FiveClassNBClassifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


def load_features():
    f = open('FiveClassFeatureSet.pickle', 'rb')
    features = pickle.load(f)
    f.close()
    return features


def extract_features(document):
    document_words = set(document)
    features = {}
    word_features = load_features()
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


if __name__ == '__main__':
    main()

