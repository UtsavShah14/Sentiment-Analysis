import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocessing_data import basic_stopwords_list


# from nltk.corpus import stopwords
# import libsvm


class TrainingModel:
    def __init__(self):
        pass

    def svm_classifier(self):
        pass

    def logr_classifier(self):
        pass


if __name__ == '__main__':
    tweet_csv = pd.read_csv("Tweets.csv")
    text_list = []
    text_sentiment_list = []
    for i, index in enumerate(tweet_csv.index):
        text_list.append(tweet_csv['Text'][index])
        text_sentiment_list.append(tweet_csv['Sentiment'][index])
    ngram_vectorizer = CountVectorizer(
        binary=True,
        ngram_range=(1, 3),
        max_features=500,
        stop_words=basic_stopwords_list
    )
    print(ngram_vectorizer.stop_words)
    X = ngram_vectorizer.fit_transform(text_list)
    # X = ngram_vectorizer.transform(text_list)
    X_train, X_val, y_train, y_val = train_test_split(
        X, text_sentiment_list, train_size=0.8
    )
    heloo = ngram_vectorizer.transform(['Hi whatsapp'])
    svm = LinearSVC(C=0.5)
    svm.fit(X_train, y_train)
    prediction = svm.predict(X_val)
    # print(prediction)
    print("Accuracy for C=0.5: %s" % (accuracy_score(y_val, prediction)))
    print(svm.predict(heloo))

# ngram_vectorizer = CountVectorizer(
#     binary=True,
#     ngram_range=(1, 2)
# )
# print("Fit-vectorize")
# ngram_vectorizer.fit(corpus_clean)
# print("Transform-vectorize")
# x = ngram_vectorizer.transform(corpus_clean)
# # # x_x_test =
# # # features = vectorizer.fit_transform(text_list)
# # # print((vectorizer.get_feature_names()))
# # # features_nd = features.toarray(dtype=object)  # for easy usage
# #
# X_train, X_test, y_train, y_test = train_test_split(
#     x,
#     text_sentiment_list,
#     train_size=0.80,
# )
# # # random_state=1234
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
#     log_model = LogisticRegression(C=c, max_iter=250)
#     log_model = log_model.fit(X=X_train, y=y_train)
#     y_pred = log_model.predict(X_test)
#     print("Accuracy for C=%s: %s" % (c, accuracy_score(y_test, y_pred)))
# # # print(accuracy_score(y_test, y_pred))

# ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=basic_stopwords_list)
# X = ngram_vectorizer.fit_transform(text_list)
# X = ngram_vectorizer.transform(text_list)
# print(X)
# X_test = ngram_vectorizer.transform(reviews_test_clean)

# X_train, X_val, y_train, y_val = train_test_split(
#     X, text_sentiment_list, train_size=0.75
# )

# ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), max_features=100, stop_words=basic_stopwords_list)
# ngram_vectorizer.fit(["I dont feel so good Mr. Strak"])
# heloo = ngram_vectorizer.transform(["I dont feel so good Mr. Strak"])

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
