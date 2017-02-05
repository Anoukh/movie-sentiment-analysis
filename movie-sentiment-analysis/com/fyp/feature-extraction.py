import nltk
import csv

word_features = []


def main():

    reviews = []

    with open("./../../Dataset/tokenizedoutputpos.csv", "rb") as f:
        reader_positive = csv.reader(f)
        positive_list = list(reader_positive)
    with open("./../../Dataset/tokenizedoutputneg.csv", "rb") as f:
        reader_negative = csv.reader(f)
        negative_list = list(reader_negative)
    # Import the Negative and Positive Sets
    for(words, sentiment) in positive_list + negative_list:
        words_filtered = [e.lower() for e in words.split() if len(e)>=3]
        reviews.append((words_filtered, sentiment))

    global word_features
    word_features = get_word_features(get_words_in_reviews(reviews))
    training_set = nltk.classify.apply_features(extract_features, reviews, labeled=True)
    print len(training_set)
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    print classifier.show_most_informative_features(60)

    tweet = 'I hate this pathetic movie'
    print classifier.classify(extract_features(tweet.split()))


def get_words_in_reviews(reviews):
    all_words =[]
    for(words, sentiment) in reviews:
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
