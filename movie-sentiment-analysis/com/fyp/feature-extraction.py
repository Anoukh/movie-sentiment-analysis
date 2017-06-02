import nltk
from nltk import FreqDist
import csv
from numpy import *
# import random

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

    with open("./../../Dataset/AssasinsCreed.csv", "rb") as f:
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

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print classifier.show_most_informative_features(10)


    # global result_one


    #To increase the accuracy, entire training set has been trained

    # global accuracy1
    # true_positive1 = 0
    # true_negative1 = 0
    # false_positive1 = 0
    # false_negative1 = 0
    # for (words, sentiment) in testing_negative + testing_positive:
    #     result_one = classifier.classify(extract_features(words.split()))
    #     if sentiment == 'positive':
    #         if result_one == 'positive':  # Real Positive. Predicted Positive
    #             true_positive1 += 1
    #         else:  # Real Positive. Predicted Negative
    #             false_negative1 += 1
    #     else:
    #         if result_one == 'negative':  # Real Negative. Predicted Negative
    #             true_negative1 += 1
    #         else:  # Real Negative. Predicted Positive
    #             false_positive1 += 1
    #
    # accuracy1 = float((true_positive1 + true_negative1) / float(len(testing_negative + testing_positive))) * 100
    # print accuracy1

    true_positive2 = 0
    true_negative2 = 0
    false_positive2 = 0
    false_negative2 = 0
    for (word, text, sentiment) in assasins_testing:
        result2 = classifier.classify(extract_features(word.split()))
        if sentiment == 'positive':
            if result2 == 'positive':  # Real Positive. Predicted Positive
                true_positive2 += 1
            else:  # Real Positive. Predicted Negative
                false_negative2 += 1
        else:
            if result2 == 'negative':  # Real Negative. Predicted Negative
                true_negative2 += 1
            else:  # Real Negative. Predicted Positive
                false_positive2 += 1

    accuracy_two = float((true_positive2 + true_negative2) / float(len(assasins_testing))) * 100
    print accuracy_two

    with open("./../../Dataset/AssasinsCreed.csv", "rb") as f:
        with open("./../../Dataset/Output/AssasinsCreed-Output.csv", "wb") as w:
            reader = csv.reader(f)
            writer = csv.writer(w)

            all = []
            for (word, text, sentiment) in assasins_testing:
                row = next(reader)
                result2 = classifier.classify(extract_features(word.split()))
                row.append(result2)
                all.append(row)

            for row in reader:
                row.append(row[0])
                all.append(row)
            writer.writerows(all)


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
