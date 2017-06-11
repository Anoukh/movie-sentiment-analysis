import unicodecsv as csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
selection = SelectPercentile(f_classif, percentile=10)
classifier = BernoulliNB()


def load_test_file():
    with open("/Users/anoukh/FYP/ToGaiya/standford-2classes-mixed_processed.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        reader.next()
        data = []
        target = []
        for row in reader:
            if row[0] and row[1]:
                data.append(row[0])
                target.append(row[1])

        data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, target,
                                                                                             test_size=0.2,
                                                                                             random_state=43)

        return data_train, data_test, target_train, target_test


def tfidf(data_train, data_test, target_train):
    global vectorizer
    data_train_transformed = vectorizer.fit_transform(data_train)
    data_test_transformed = vectorizer.transform(data_test)

    global selection
    selection.fit(data_train_transformed, target_train)
    data_train_transformed = selection.transform(data_train_transformed).toarray()
    data_test_transformed = selection.transform(data_test_transformed).toarray()

    return data_train_transformed, data_test_transformed


def classify(data_train_transformed, data_test_transformed, target_train, target_test):
    global classifier
    classifier.fit(data_train_transformed, target_train)
    predicted = classifier.predict(data_test_transformed)
    evaluate_model(target_test, predicted)


def evaluate_model(target_true, target_predicted):
    print classification_report(target_true, target_predicted)
    print "The accuracy score is {:.2%}".format(accuracy_score(target_true, target_predicted))


def find_sentiment(text):
    global vectorizer
    global selection
    global classifer
    transformed_text = vectorizer.transform(text)
    transformed_text = selection.transform(transformed_text).toarray()
    print classifier.predict(transformed_text)


def main():
    data_train, data_test, target_train, target_test = load_test_file()
    data_train_transformed, data_test_transformed = tfidf(data_train, data_test, target_train)
    classify(data_train_transformed, data_test_transformed, target_train, target_test)
    find_sentiment(["Anoukh is an awesome good movie", "Anoukh is horrible movie"])


if __name__ == '__main__':
    main()
