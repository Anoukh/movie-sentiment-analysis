import nltk
from nltk import FreqDist
import csv
import shutil
import random

word_features = []


def main():
    training_reviews = []

    with open("./../../Dataset/tokenizedoutputpos.csv", "rb") as f:
        reader_positive = csv.reader(f)
        positive_list = list(reader_positive)
        # random.shuffle(positive_list)
        # training_positive = positive_list[:(len(positive_list) * 70 / 100)]
        # testing_positive = positive_list[(len(positive_list) * 70 / 100):]

    with open("./../../Dataset/tokenizedoutputneg.csv", "rb") as f:
        reader_negative = csv.reader(f)
        negative_list = list(reader_negative)
        # random.shuffle(negative_list)
        # training_negative = negative_list[: len(negative_list) * 70 / 100]
        # testing_negative = negative_list[(len(negative_list) * 70 / 100):]

    with open("./../../Dataset/Output/AC_freqdist_nb_Output.csv", "rb") as f:
        reader_testing = csv.reader(f)
        assasins_testing = list(reader_testing)

    # Import the Negative and Positive Sets
    for (words, sentiment) in positive_list + negative_list:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        training_reviews.append((words_filtered, sentiment))

    global word_features
    word_features = get_word_features(get_words_in_reviews(training_reviews))
    training_set = nltk.classify.apply_features(extract_features, training_reviews, labeled=True)
    print len(training_set)

    classifier = nltk.MaxentClassifier.train(training_set, max_iter=1)

    print classifier.show_most_informative_features(10)

    global predicted_result
    global actual_result

    # actual_result = [sentiment for (words, sentiment) in testing_negative + testing_positive]
    with open("./../../Dataset/Output/AC_freqdist_nb_Output.csv", "rb") as f:
        with open("./../../Dataset/Output/AC_freqdist_nb_Output2.csv", "wb") as w:
            reader = csv.reader(f)
            writer = csv.writer(w)

            all = []
            for (word, text, sentiment, predicted_nb) in assasins_testing:
                row = next(reader)
                predicted_result = classifier.classify(extract_features(word.split()))
                row.append(predicted_result)
                all.append(row)

            for row in reader:
                row.append(row[0])
                all.append(row)
            writer.writerows(all)

        shutil.move("./../../Dataset/Output/AC_freqdist_nb_Output2.csv", "./../../Dataset/Output/AC_freqdist_nb_Output.csv")


def get_words_in_reviews(reviews):
    all_words = []
    for (words, sentiment) in reviews:
        all_words.extend(words)
        return all_words


def get_word_features(word_list):
    word_list = FreqDist(word_list)
    global word_features
    word_features = word_list.keys()
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


if __name__ == '__main__':
    main()
