import preprocessing_data
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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
            sublinear_tf=True,
            use_idf=True,
        )
        self.corpus = self.vectorizer.fit_transform(self.clean_text_list)


def log_r_model_run(x_train, y_train):
    t0 = time.time()
    model = LogisticRegression()  # The logistic regression model
    solvers = ['newton-cg', 'lbfgs', 'liblinear']  # Algorithm to use in the optimization
    penalty = ['l2']  # norms of penalization
    C = [100, 10, 1.0, 0.1, 0.01]  # Regularization parameter, small values is stronger regularization
    max_iter = [1000]  # Number of iterations for convergence. Can use higher values, time increases exponentially

    grid = dict(solver=solvers, C=C, penalty=penalty, max_iter=max_iter)
    cv = RepeatedStratifiedKFold(
        n_splits=10,  # Number of folds, small value is proportional to time an inversely proportional to accuracy
        n_repeats=3,
        random_state=1
    )
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,  # For parallel processing
        cv=cv,
        scoring='accuracy',  # Metric to find the best model
        error_score=0,
        refit=True
    )
    grid_result = grid_search.fit(x_train, y_train)  # Fitting the model
    # print("Time taken to return the best fit LR model: ", time.time() - t0)
    return grid_result.best_estimator_  # Return the bestfit model


if __name__ == '__main__':
    train = TrainingModel()

    x_training, x_validation, y_training, y_validation = train_test_split(
        train.corpus,
        train.text_sentiment_list,
        train_size=0.8
    )

    t0 = time.time()
    model = LogisticRegression()  # The logistic regression model
    solvers = ['newton-cg', 'lbfgs', 'liblinear']  # Algorithm to use in the optimization
    penalty = ['l2']  # norms of penalization
    C = [100, 10, 1.0, 0.1, 0.01]  # Regularization parameter, small values is stronger regularization
    max_iter = [1000]  # Number of iterations for convergence. Can use higher values, time increases exponentially

    grid = dict(solver=solvers, C=C, penalty=penalty, max_iter=max_iter)
    cv = RepeatedStratifiedKFold(
        n_splits=10,  # Number of folds, small value is proportional to time an inversely proportional to accuracy
        n_repeats=3,
        random_state=1
    )
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,  # For parallel processing
        cv=cv,
        scoring='accuracy',  # Metric to find the best model
        error_score=0,
        refit=False
    )
    grid_result = grid_search.fit(x_training, y_training)
    print("Summarizing the results for our  SVM Classifier: ")
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Took ", time.time() - t0, "to finish the process")
