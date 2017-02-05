import nltk
import csv


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

    word_features = get_word_features(get_words_in_reviews(reviews))
    print len(word_features)
    print word_features


def get_words_in_reviews(reviews):
    all_words =[]
    for(words, sentiment) in reviews:
        all_words.extend(words)
        return all_words


def get_word_features(word_list):
    word_list = nltk.FreqDist(word_list)
    word_features = word_list.keys()
    return word_features

if __name__ == '__main__':
    main()
