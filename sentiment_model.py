import time
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import preprocessing_data
import NaiveBayes_Model as nb
import LogisticRegression_Model as lr
import SupportVectorMachine_model as s_v_m


class TrainingModel:
    """
    The class initializes and loads the data from CSV into a list and initializes vector to convert the data.
    The data is preprocessed and converted to list
    This list is vectorized using the vectorizer.
    This class has our classifiers methods that calls our model files for Training the models
    """
    def __init__(self):
        tweet_csv = pd.read_csv("Tweets.csv")
        self.text_list = []
        self.text_sentiment_list = []
        self.clean_text_list = []
        for i, index in enumerate(tweet_csv.index):
            self.text_list.append(tweet_csv['Text'][index])
            self.text_sentiment_list.append(tweet_csv['Sentiment'][index])

        for i, text in enumerate(self.text_list):
            self.clean_text_list.append(preprocessing_data.get_clean_text(text))

        self.vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
            use_idf=True
        )
        self.corpus = self.vectorizer.fit_transform(self.clean_text_list)
        joblib.dump(self.vectorizer, 'vectorizer.sav')

    @staticmethod
    # SVM Classifier method
    def svm_classifier(x_train, x_val, y_train, y_val):
        t0 = time.time()
        # Calls svm_model_run from SupportVectorMachine_model.py file.
        svm_model = s_v_m.svm_model_run(x_train, y_train)  # Returns the best fit classifier
        svm_model_fit = svm_model.fit(x_train, y_train)  # Fitting our dataset
        prediction = svm_model_fit.predict(x_val)  # Predicting with test values
        train.get_score('SVM', svm_model_fit, prediction, y_val)  # Getting the score for overview
        print("Time taken to Train SVM: " + str(time.time() - t0))
        # For further comparison between all models
        weighted_f1 = metrics.f1_score(y_val, prediction, average='weighted')
        return 'Support Vector Machine', svm_model_fit, weighted_f1

    @staticmethod
    # NB Classifier method
    def naive_bayes(x_train, x_val, y_train, y_val):
        t0 = time.time()
        # Calls nb_model_run from NaiveBayes_model.py file.
        nb_model = nb.nb_model_run(x_train, y_train)  # Returns the best fit classifier
        nb_model_fit = nb_model[0].fit(x_train, y_train)  # Fitting our dataset
        prediction = nb_model_fit.predict(x_val)  # Predicting with test values
        train.get_score('Naive Bayes', nb_model_fit, prediction, y_val)  # Getting the score for overview
        print("Time taken to Train NaiveBayes: " + str(time.time() - t0))
        # For further comparison between all models
        weighted_f1 = metrics.f1_score(y_val, prediction, average='weighted')
        return 'Naive Bayes', nb_model_fit, weighted_f1

    @staticmethod
    # LogR Classifier method
    def logr_classifier(x_train, x_val, y_train, y_val):
        t0 = time.time()
        # Calls log_r_model_run from LogisticRegression_model.py file.
        lr_model = lr.log_r_model_run(x_train, y_train)  # Returns the best fit classifier
        lr_model_fit = lr_model.fit(x_train, y_train)  # Fitting our dataset
        prediction = lr_model_fit.predict(x_val)  # Predicting with test values
        train.get_score('LogisticRegression', lr_model_fit, prediction, y_val)  # Getting the score for overview
        print("Time taken to Train LogR: " + str(time.time() - t0))
        # For further comparison between all models
        weighted_f1 = metrics.f1_score(y_val, prediction, average='weighted')
        return 'Logistic Regression', lr_model_fit, weighted_f1

    @staticmethod
    # This method gets the score of the classifier passed
    # It will return a plot showing the confusion matrix
    def get_score(classifier_name, classifier, prediction, y_val):
        print("Stats for " + classifier_name + " : ")
        print('\tClassification report\n', str(metrics.classification_report(y_val, prediction)))
        disp = metrics.plot_confusion_matrix(
            classifier,
            x_validation,
            y_validation,
            # display_labels=train.text_sentiment_list,
            cmap=plt.cm.Blues
            )
        disp.ax_.set_title(classifier_name)
        plt.show(block=False)


if __name__ == '__main__':
    # Stores scores of best fit models along with the classifier instances as a list of tuple
    selected_model = []  # Will compare the models with each other
    train = TrainingModel()

    # Splitting the dataset into training and testing (validation) data
    x_training, x_validation, y_training, y_validation = train_test_split(
        train.corpus,
        train.text_sentiment_list,
        train_size=0.9
    )

    model_name, clf, weighted_f1_score = train.naive_bayes(x_training, x_validation, y_training, y_validation)
    selected_model.append((model_name, clf, weighted_f1_score))
    model_name, clf, weighted_f1_score = train.logr_classifier(x_training, x_validation, y_training, y_validation)
    selected_model.append((model_name, clf, weighted_f1_score))
    model_name, clf, weighted_f1_score = train.svm_classifier(x_training, x_validation, y_training, y_validation)
    selected_model.append((model_name, clf, weighted_f1_score))

    default_score = 0
    final_model = ()
    # Compares the scores of all the models and pickles the best fit model
    for name, model, score in selected_model:
        if score * 100 > default_score:
            default_score = score * 100
            final_model = (name, model)

    print("Selected model is: ", final_model)  # Print the final selected model
    joblib.dump(final_model[1], 'final_model.sav')  # Pickle the model
