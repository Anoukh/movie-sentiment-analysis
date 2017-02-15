import nltk
from nltk import FreqDist
import csv
import random

word_features = []


def main():
    training_reviews = []

    with open("./../../Dataset/tokenizedoutputpos.csv", "rb") as f:
        reader_positive = csv.reader(f)
        positive_list = list(reader_positive)
        random.shuffle(positive_list)
        training_positive = positive_list[:(len(positive_list) * 70 / 100)]
        testing_positive = positive_list[(len(positive_list) * 70 / 100):]

    with open("./../../Dataset/tokenizedoutputneg.csv", "rb") as f:
        reader_negative = csv.reader(f)
        negative_list = list(reader_negative)
        random.shuffle(negative_list)
        training_negative = negative_list[: len(negative_list) * 70 / 100]
        testing_negative = negative_list[(len(negative_list) * 70 / 100):]

    # Import the Negative and Positive Sets
    for (words, sentiment) in training_positive + training_negative:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        training_reviews.append((words_filtered, sentiment))

    global word_features
    word_features = get_word_features(get_words_in_reviews(training_reviews))
    training_set = nltk.classify.apply_features(extract_features, training_reviews, labeled=True)
    print len(training_set)

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print classifier.show_most_informative_features(10)

    # tweet = 'I hate this pathetic movie'
    global result
    global accuracy
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for (words, sentiment) in testing_negative + testing_positive:
        result = classifier.classify(extract_features(words.split()))
        if sentiment == 'positive':
            if result == 'positive':  # Real Positive. Predicted Positive
                true_positive += 1
            else:  # Real Positive. Predicted Negative
                false_negative += 1
        else:
            if result == 'negative':  # Real Negative. Predicted Negative
                true_negative += 1
            else:  # Real Negative. Predicted Positive
                false_positive += 1

    accuracy = float((true_positive + true_negative) / float(len(testing_negative + testing_positive)))*100
    print accuracy


def get_words_in_reviews(reviews):
    all_words = []
    for (words, sentiment) in reviews:
        all_words.extend(words)
        return all_words


def get_word_features(word_list):
    word_list = nltk.FreqDist(word_list)
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
