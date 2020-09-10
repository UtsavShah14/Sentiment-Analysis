import preprocessing_data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import time


class TrainingModel:
    """
    The class initializes and loads the data from CSV into a list and initializes vector to convert the data.
    The data is preprocessed and converted to list
    This list is vectorized using the vectorizer.
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
            use_idf=True,
        )
        self.corpus = self.vectorizer.fit_transform(self.clean_text_list)


def nb_model_run(x_train, y_train):
    t0 = time.time()
    best_models = []  # For the bestfit model from all the Naive Bayes classifiers
    # Three Naive Bayes models initialized
    model_list = [
        BernoulliNB(),
        MultinomialNB(),
        ComplementNB(),
    ]

    alpha = [50, 10, 1.0, 0.1, 0.01, 1e-3, 1e-4]  # Smoothening Parameter
    grid = dict(alpha=alpha)
    cv = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=1
    )
    for model in model_list:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            n_jobs=-1,
            cv=cv,
            scoring='accuracy',
            error_score=0,
            refit=True
        )
        grid_result = grid_search.fit(x_train, y_train)
        best_models.append((grid_result.best_estimator_, model, grid_result.best_score_))
        # print(grid_result.estimator, ": ", grid_result.best_score_, grid_result.best_params_)
    final_model = ()  # Chosen best classifier
    default_score = 0
    # Comparing the score to find the best of the three Naive Bayes classifiers
    for estimator, model, score in best_models:
        if score * 100 > default_score:
            default_score = score * 100
            final_model = (estimator, model, score)
    # print("Time taken to return the best fit NB model: ", time.time() - t0)
    return final_model  # Returns the best classifier


if __name__ == '__main__':
    train = TrainingModel()

    x_training, x_validation, y_training, y_validation = train_test_split(
        train.corpus,
        train.text_sentiment_list,
        train_size=0.8
    )
    t0 = time.time()
    best_models = []  # For the bestfit model from all the Naive Bayes classifiers
    # Three Naive Bayes models initialized
    model_list = [
        BernoulliNB(),
        MultinomialNB(),
        ComplementNB(),
    ]

    alpha = [50, 10, 1.0, 0.1, 0.01, 1e-3, 1e-4]  # Smoothening Parameter
    grid = dict(alpha=alpha)
    cv = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=1
    )
    for model in model_list:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid,
            n_jobs=-1,  # For parallel processing
            cv=cv,
            scoring='accuracy',  # Metric to find the best model
            error_score=0,
            refit=True
        )
        grid_result = grid_search.fit(x_training, y_training)
        print("Summarizing results for ", model, " Classifier: ")
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        print()
