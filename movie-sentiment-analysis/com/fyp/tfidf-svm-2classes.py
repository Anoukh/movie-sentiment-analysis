from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import unicodecsv as csv
import numpy

from sklearn.model_selection import train_test_split

# X, Y = make_multilabel_classification(n_samples=5, n_features=10, n_classes=5, n_labels=2, length=50,
#                                       allow_unlabeled=False, sparse=False, return_indicator='dense',
#                                       return_distributions=False, random_state=None)
#
# print X
# print "~~~~~~~"
# Y = np.array([[1, 0, 0, 0, 0],
#      [0, 1, 0, 0, 0],
#      [0, 0, 1, 0, 0],
#      [0, 0, 0, 1, 0],
#      [0, 0, 0, 0, 1]])
#
# print Y
# classif = OneVsRestClassifier(SVC(kernel='linear'))
# classif.fit(X, Y)
#
# print "~~~~~~~"
# # print classif
# J = [[5.,   0.,   7.,   4.,   6.,   3.,   1.,   5.,   2.,   8.],
# [8.,   2.,   7.,   4.,   6.,  10.,   2.,   4.,   8.,   8.],
# [8.,   8.,   3.,   7.,   3.,   6.,   4.,   0.,   7.,   3.],
# [6.,   0.,   5.,   1.,  15.,   3.,   5.,   0.,   9.,   2.],
# [10.,   0.,  14.,   2.,  10.,   1.,   2.,   2.,   6.,   4.]]
# print classif.predict(J)


def load_test_file():
    with open("F:/FYP/Repo/movie-sentiment-analysis/Dataset/Stanford/standford-2classes-mixed_processed.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        data = []
        target = []
        for row in reader:
            if row[1] and row[0]:
                data.append(row[1])
                target.append(row[0])

        training_data = data[:len(data) * 70 / 100]
        testing_data = data[len(data) * 70 / 100:]
        training_target = target[:len(target) * 70 / 100]
        testing_target = target[len(target) * 70 / 100:]

        return training_data, training_target, testing_data, testing_target


def pre_process(training_data, training_target, testing_data, testing_target):

    new_target = []
    new_test = []
    # count_vectorizer = CountVectorizer(binary='true')
    # data = count_vectorizer.fit_transform(data)
    # tfidf_data = TfidfTransformer(use_idf=True).fit_transform(data)

    vectorizer = TfidfVectorizer(analyzer='word', use_idf=True, max_features=2000).fit(training_data+testing_data)
    tfidf_data = vectorizer.transform(training_data)
    tfidf_testing = vectorizer.transform(testing_data)

    for sentiment in training_target:
        if sentiment == "positive":
            new_target.append([1, 0])
        elif sentiment == "negative":
            new_target.append([0, 1])

    for sentiment in testing_target:
        if sentiment == "positive":
            new_test.append([1, 0])
        elif sentiment == "negative":
            new_test.append([0, 1])

    numpy_target = np.array(new_target)
    numpy_test = np.array(new_test)
    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(tfidf_data, numpy_target)

    result = classif.predict(tfidf_testing)

    trueValue = 0
    falseValue = 0

    if len(result) == len(numpy_test):
        print "Same"
        for i in range(0, len(result)):
            temp = (result[i] == numpy_test[i])
            if max(temp) == False or min(temp) == False:
                falseValue += 1
            else:
                trueValue += 1
    print str(trueValue) + " " + str(falseValue)
    print "Accuracy = " + str(trueValue*100./(trueValue+falseValue))


def main():
    training_data, training_target, testing_data, testing_target = load_test_file()
    pre_process(training_data, training_target, testing_data, testing_target)


if __name__ == '__main__':
    main()
