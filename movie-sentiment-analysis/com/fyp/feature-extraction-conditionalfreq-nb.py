import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist


POLARITY_DATASET = os.path.join('F:/FYP/Repo/movie-sentiment-analysis', 'Dataset')
POSITIVE_FILE = os.path.join(POLARITY_DATASET, 'positiveset.txt')
NEGATIVE_FILE = os.path.join(POLARITY_DATASET, 'negativeset.txt')


def evaluate_features(feature_select):
    posFeatures = []
    negFeatures = []

    with open(POSITIVE_FILE, 'r') as posSentences:
        for i in posSentences:
            posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)

    with open(NEGATIVE_FILE, 'r') as negSentences:
        for i in negSentences:
            negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)

    posCutoff = int(math.floor(len(posFeatures)*70/100))
    negCutoff = int(math.floor(len(negFeatures)*70/100))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    classifier = NaiveBayesClassifier.train(trainFeatures)

    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    # print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    # print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    # print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    # print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10)


def make_full_dict(words):
    return dict([(word, True) for word in words])

print 'using all words as features'

evaluate_features(make_full_dict)


def create_word_scores():
    posWords = []
    negWords = []

    with open(POSITIVE_FILE,'r') as posSentences:
        for i in posSentences:
            posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            posWords.append(posWord)

    with open(NEGATIVE_FILE, 'r') as negSentences:
        for i in negSentences:
            negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            negWords.append(negWord)

    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()

    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores

word_scores = create_word_scores()


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

numbers_to_test = [10, 100, 1000, 10000, 15000, 20000]
for num in numbers_to_test:
    print 'evaluating best %d word features' % (num)
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)




