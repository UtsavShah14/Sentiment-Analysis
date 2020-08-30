import time

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class TrainingModel:
    def __init__(self):
        tweet_csv = pd.read_csv("Tweets.csv")
        self.text_list = []
        self.text_sentiment_list = []
        for i, index in enumerate(tweet_csv.index):
            self.text_list.append(tweet_csv['Text'][index])
            self.text_sentiment_list.append(tweet_csv['Sentiment'][index])

        self.vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
            use_idf=True
        )
        self.corpus = self.vectorizer.fit_transform(self.text_list)
        joblib.dump(self.vectorizer, 'vectorizer.sav')

    @staticmethod
    def svm_classifier(x_train, x_val, y_train, y_val):
        t0 = time.time()
        svm_classifier = svm.SVC(
            kernel='linear',
            C=0.05
        )
        svm_classifier.fit(x_train, y_train)
        joblib.dump(svm_classifier, 'svm_classifier.sav')
        prediction = svm_classifier.predict(x_val)
        t1 = time.time()
        train.get_score('SVM', prediction, y_val)
        print("Time taken to Train SVM: " + str(t1 - t0))

    @staticmethod
    def logr_classifier(x_train, x_val, y_train, y_val):
        t0 = time.time()
        log_model = LogisticRegression(
            penalty='l2',
            C=0.5,
            max_iter=250,
            solver='saga'
        )
        log_model.fit(x_train, y_train)
        joblib.dump(log_model, 'logr_classifier.sav')
        prediction = log_model.predict(x_val)
        t1 = time.time()
        train.get_score('LogisticRegression', prediction, y_val)
        print("Time taken to Train LogR: " + str(t1 - t0))

    @staticmethod
    def get_score(classifier_name, prediction, y_val):
        print("Stats for " + classifier_name + " : ")
        print("\tAccuracy: " + str(metrics.accuracy_score(y_val, prediction)))
        print("\tPrecision:", str(metrics.precision_score(y_val, prediction, average='weighted')))
        print("\tRecall:", metrics.recall_score(y_val, prediction, average='weighted'))
        print("\tF1 score: ", str(metrics.f1_score(y_val, prediction, average='weighted')))


if __name__ == '__main__':
    train = TrainingModel()
    x_training, x_validation, y_training, y_validation = train_test_split(
        train.corpus,
        train.text_sentiment_list,
        train_size=0.8
    )

    train.logr_classifier(x_training, x_validation, y_training, y_validation)
    train.svm_classifier(x_training, x_validation, y_training, y_validation)
