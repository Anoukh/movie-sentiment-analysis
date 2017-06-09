import nltk
from nltk import FreqDist
import csv
from numpy import *
import random
import pickle


word_features = []


def main():
    training_reviews = []

    with open("./../../Dataset/Stanford/stanford_annotated_processed.csv", "rb") as f:
        reader_tweets = csv.reader(f)
        tweet_list = list(reader_tweets)
       #random.shuffle(tweet_list)
        train_set = tweet_list[:(len(tweet_list) * 70 / 100)]
        test_set = tweet_list[(len(tweet_list) * 70 / 100):]

    for (words, sentiment) in train_set:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        training_reviews.append((words_filtered, sentiment))

    global word_features
    word_features = get_word_features(get_words_in_reviews(training_reviews))
    training_set = nltk.classify.apply_features(extract_features, training_reviews, labeled=True)
    print len(training_set)

    classifier = nltk.NaiveBayesClassifier.train(training_set)

    f = open('FiveClassNBClassifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

    print classifier.show_most_informative_features(10)

    true_vp = 0
    true_p = 0
    true_neu = 0
    true_neg = 0
    true_vneg = 0
    other = 0
    for (word, sentiment) in test_set:
        result = classifier.classify(extract_features(word.split()))
        if sentiment == 'very_positive':
            if result == 'very_positive':
                true_vp += 1
        elif sentiment == 'positive':
            if result == 'positive':
                true_p += 1
        elif sentiment == 'neutral':
            if result == 'neutral':
                true_neu += 1
        elif sentiment == 'negative':
            if result == 'negative':
                true_neg += 1
        elif sentiment == 'very negative':
            if result == 'very negative':
                true_vneg += 1
        else:
            other += 1

    accuracy = float((true_vp + true_p + true_neu + true_neg + true_vneg) / float(len(test_set))) * 100
    print accuracy

    # with open("./../../Dataset/logan.csv", "rb") as f:
    #     with open("./../../Dataset/Output/logan-output.csv", "wb") as w:
    #         reader = csv.reader(f)
    #         writer = csv.writer(w)
    #
    #         all = []
    #         for (word, sentiment) in test_set:
    #             row = next(reader)
    #             result2 = classifier.classify(extract_features(word.split()))
    #             row.append(result2)
    #             all.append(row)
    #
    #         for row in reader:
    #             row.append(row[0])
    #             all.append(row)
    #         writer.writerows(all)


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

    f = open('FiveClassFeatureSet.pickle', 'wb')
    pickle.dump(word_features, f)
    f.close()
    return features


if __name__ == '__main__':
    main()












