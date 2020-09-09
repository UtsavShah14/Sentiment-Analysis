import preprocessing_data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class TrainingModel:
    def __init__(self):
        tweet_csv = pd.read_csv("Tweets.csv")
        self.text_list = []
        self.text_sentiment_list = []
        self.clean_text_list = []
        for i, index in enumerate(tweet_csv.index):
            self.text_list.append(tweet_csv['Text'][index])
            self.text_sentiment_list.append(tweet_csv['Sentiment'][index])
            # print(self.text_list[i])

        for i, text in enumerate(self.text_list):
            self.clean_text_list.append(preprocessing_data.get_clean_text(text))
            # print(self.clean_text_list[i])
            # print(text)
            # time.sleep(2)

        self.vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
            use_idf=True,
        )
        self.corpus = self.vectorizer.fit_transform(self.clean_text_list)
        # joblib.dump(self.vectorizer, 'vectorizer.sav')


if __name__ == '__main__':
    train = TrainingModel()
    nb_Bnb = BernoulliNB()
    nb_Catnb = CategoricalNB()
    nb_Comnb = ComplementNB()
    nb_Mnb = MultinomialNB()
    model_list = [
        BernoulliNB(),
        ComplementNB(),
        MultinomialNB()
        ]
    x_training, x_validation, y_training, y_validation = train_test_split(
        train.corpus,
        train.text_sentiment_list,
        train_size=0.8
    )
    score = ['accuracy', 'f1_weighted']
    alpha = [50, 10, 1.0, 0.1, 0.01, 1e-3, 1e-4]
    # define grid search
    grid = dict(alpha=alpha)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    for model in model_list:
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=grid,
                                   n_jobs=-1,
                                   cv=cv,
                                   scoring='accuracy',
                                   error_score=0,
                                   refit=False)
        grid_result = grid_search.fit(x_training, y_training)
        # summarize results
        print("Summarizing results for ", model, " Classifier: ")
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        print()