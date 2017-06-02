import csv
import random
import nltk
import collections
import nltk.metrics
import itertools
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder
from nltk import precision


'''
Training a model using naive bayes and classify data
'''
word_features = []


def main():

    fp = open("./../../Dataset/libsvm.csv", "rb")
    reader = csv.reader(fp, delimiter=',')
    raw_tweets = []
    train_tweets = []
    for row in reader:
        raw_tweets.append((row[0], row[1]))

    random.shuffle(raw_tweets)
    v_train = raw_tweets[:len(raw_tweets)*75/100]
    v_test = raw_tweets[len(raw_tweets)*75/100:]

    for (tweet, sentiment) in v_train:
        words_filtered = []
        for word in word_tokenize(tweet):
            words_filtered.append(word)
        train_tweets.append((words_filtered, sentiment))

    global word_features
    word_features=get_word_features(get_words(train_tweets))

    training_set = nltk.classify.apply_features(bigram_word_feats, train_tweets, labeled=True)
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print classifier.show_most_informative_features(10)

    true_vp = 0
    true_p = 0
    true_neu = 0
    true_neg = 0
    true_vneg = 0
    other = 0
    for (word, sentiment) in v_test:
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

    accuracy = float((true_vp + true_p + true_neu + true_neg + true_vneg) / float(len(v_test))) * 100
    print accuracy


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def get_words(tweet_set):
    all_words = []
    for (tweet, sentiment) in tweet_set:
        all_words.extend(tweet)
    return all_words


def get_word_features(all_words):
    word_freq = FreqDist(all_words)
    word_features = word_freq.keys()
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# def create_confussion_matrix(v_test, classifier):
#     test_truth = [s for (t, s) in v_test]
#     test_predict = [classifier.classify(extract_features(t.split())) for (t, s) in v_test]
#     confussion_matrix = nltk.ConfusionMatrix(test_truth, test_predict)
#     print 'Confusion Matrix'
#     print(confussion_matrix)
#     accuracy = float(confussion_matrix.correct/confussion_matrix.total)
#     print("Accuracy:", accuracy)


if __name__ == '__main__':
    main()
