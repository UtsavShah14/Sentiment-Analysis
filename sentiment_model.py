# import pickle
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocessing_data import basic_stopwords_list


# import libsvm


class TrainingModel:
    def __init__(self):
        tweet_csv = pd.read_csv("Tweets.csv")
        self.text_list = []
        self.text_sentiment_list = []
        for i, index in enumerate(tweet_csv.index):
            self.text_list.append(tweet_csv['Text'][index])
            self.text_sentiment_list.append(tweet_csv['Sentiment'][index])

        self.ngram_vectorizer = CountVectorizer(
            binary=True,
            ngram_range=(1, 3),
            max_features=500,
            stop_words=basic_stopwords_list
        )

        self.corpus = self.ngram_vectorizer.fit_transform(self.text_list)
        # pickle.dump(self.ngram_vectorizer, open('vectorizer.sav', 'wb'))
        joblib.dump(self.ngram_vectorizer, 'vectorizer.sav')
        # corpus = ngram_vectorizer.fit_transform(self.text_list)

    def preprocessor(self):
        # corpus = self.ngram_vectorizer.fit_transform(self.text_list)

        x_train, x_val, y_train, y_val = train_test_split(
            self.corpus, train.text_sentiment_list, train_size=0.8
        )
        return x_train, x_val, y_train, y_val

    @staticmethod
    def svm_classifier(x_train, x_val, y_train, y_val):
        svm_classifier = svm.SVC(
            kernel='linear',
            C=0.05)
        svm_classifier.fit(x_train, y_train)
        prediction = svm_classifier.predict(x_val)
        print("Accuracy for SVM: " + str(metrics.accuracy_score(y_val, prediction)))
        # print("Precision:", metrics.precision_score(y_val, prediction))
        # print("Recall:", metrics.recall_score(y_val, prediction))
        # pickle.dump(svm_classifier, open('svm_classifier.sav', 'wb'))
        joblib.dump(svm_classifier, 'svm_classifier.sav')
        # return svm_classifier

    @staticmethod
    def logr_classifier(x_train, x_val, y_train, y_val):
        log_model = LogisticRegression(C=0.5, max_iter=250)
        log_model.fit(x_train, y_train)
        prediction = log_model.predict(x_val)
        print("Accuracy for LR: " + str(metrics.accuracy_score(y_val, prediction)))
        # print("Precision:", metrics.precision_score(y_val, prediction))
        # print("Recall:", metrics.recall_score(y_val, prediction))


if __name__ == '__main__':
    train = TrainingModel()
    x_training, x_validation, y_training, y_validation = train.preprocessor()
    train.svm_classifier(x_training, x_validation, y_training, y_validation)
    train.logr_classifier(x_training, x_validation, y_training, y_validation)

    # test = train.ngram_vectorizer.transform(['This shit  is insane'])
    # print(svm_model.predict(test))
    # tweet_csv = pd.read_csv("Tweets.csv")
    # text_list = []
    # text_sentiment_list = []
    # for i, index in enumerate(tweet_csv.index):
    #     text_list.append(tweet_csv['Text'][index])
    #     text_sentiment_list.append(tweet_csv['Sentiment'][index])

    # ngram_vectorizer = CountVectorizer(
    #     binary=True,
    #     ngram_range=(1, 3),
    #     max_features=500,
    #     stop_words=basic_stopwords_list
    # )
    #
    # X = ngram_vectorizer.fit_transform(train.text_list)
    #
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, train.text_sentiment_list, train_size=0.8
    # )

    # svm = LinearSVC(C=0.05)
    # svm.fit(X_train, y_train)
    # prediction = svm.predict(X_val)
    # print(prediction)
    # print("Accuracy: " + str(accuracy_score(y_val, prediction)))
    # print(svm.predict(heloo))

# for c in [0.01, 0.05, 0.25, 0.5, 1]:
#     log_model = LogisticRegression(C=c, max_iter=250)
#     log_model = log_model.fit(X=X_train, y=y_train)
#     y_pred = log_model.predict(X_test)
#     print("Accuracy for C=%s: %s" % (c, accuracy_score(y_test, y_pred)))
# # # print(accuracy_score(y_test, y_pred))

# print(X_val)
# for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
#     svm = LinearSVC(C=c)
#     svm.fit(X_train, y_train)
#     print("Accuracy for C=%s: %s"
#           % (c, accuracy_score(y_val, svm.predict(X_val))))

# svm = LinearSVC(C=0.5)
# svm.fit(X_train, y_train)
# prediction = svm.predict(X_val)
# print(prediction)
# print("Accuracy for C=0.5: %s" % (accuracy_score(y_val, prediction)))
# confusion_matrix(y_true=y_val, y_pred=prediction, labels=['positive', 'negative'])
# print(confusion_matrix)
# dump(svm, 'SVM_classifier')
# disp = plot_confusion_matrix(svm, X_val, y_val,
#                              display_labels=['positive', 'negative', 'neutral'],
#                              cmap=plt.cm.Blues,
#                              )
# plt.show()
