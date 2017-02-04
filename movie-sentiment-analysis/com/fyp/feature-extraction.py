import nltk


def main():
    pos_reviews = []
    neg_reviews = []
    reviews = []

    # Import the Negative and Positive Sets

    for(words, sentiment) in pos_reviews + neg_reviews:
        words_filtered = [e.lower() for e in words.split() if len(e)>=1]
        reviews.append(words_filtered, sentiment)

    word_features = get_word_features(get_words_in_reviews(reviews))


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