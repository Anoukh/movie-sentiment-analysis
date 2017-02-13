import csv


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
# import numpy as np
from sklearn.metrics import accuracy_score


def load_file():
    with open("./../../Dataset/merged.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        reader.next()
        data = []
        target = []
        for row in reader:
            if row[0] and row[1]:
                data.append(row[0])
                target.append(row[1])
        return data, target


def pre_process(data, target):
    # data, target = load_file()
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    return tfidf_data


def learn_model(data, target):
    data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, target, test_size=0.3, random_state=43)
    classifier = BernoulliNB().fit(data_train, target_train)
    predicted = classifier.predict(data_test)
    evaluate_model(target_test, predicted)


def evaluate_model(target_true, target_predicted):
    print classification_report(target_true, target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true, target_predicted))


def main():
    data, target = load_file()
    tf_idf = pre_process(data, target)
    learn_model(tf_idf, target)

main()


