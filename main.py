# import os
import time

import flask as f
import joblib
import pandas as pd
import plotly.express as px
import tweepy as tw

import Access_Keys

# basic_stopwords_list = {
#     'a', 'an', 'all', 'and', 'are', 'as', 'at',
#     'be', 'but', 'can', 'do', 'did', 'for',
#     'get', 'give' 'has', 'had', 'have', 'how',
#     'i', 'if', 'in', 'is', 'it',
#     'me', 'my', 'no',
#     'of', 'on', 'or',
#     'that', 'the', 'there' 'this', 'to', 'up',
#     'was', 'we', 'what', 'when', 'why', 'where', 'would', 'with', 'will',
#     'you'
# }
#

# tweet_csv = pd.read_csv("test_data.csv")
# text_list = []
# text_sentiment_list = []

# for i, index in enumerate(tweet_csv.index):
#     text_list.append(tweet_csv['Text'][index])

# ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=basic_stopwords_list)
# X = ngram_vectorizer.fit_transform(text_list)

# vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
# vectorizer = joblib.load('vectorizer.sav')
# vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=basic_stopwords_list)
# ngram_vectorizer.fit(["I dont feel so good Mr. Stark"])
# X = vectorizer.transform(text_list)
# print(X)
# svm = pickle.load(open('svm_classifier.sav', 'rb'))
# svm = joblib.load('svm_classifier.sav')
# print(svm.predict(X))

# print(tw.__version__)
consumer_API_key = Access_Keys.consumer_API_key
consumer_API_secret_key = Access_Keys.consumer_API_secret_key
access_token = Access_Keys.access_token
access_token_secret = Access_Keys.access_token_secret
#
app = f.Flask(__name__)
app.secret_key = 'hello'


@app.route('/')
def index():
    # username = ""
    prediction = 'False'
    if 'username' in f.session:
        template_values = {
            'username': f.session['username'],
            'prediction': prediction
        }
        return f.render_template('index.html', **template_values)
    return "You are not logged in, please login in to continue.<br><a href = '/login'>" + "click here to login</a>"


@app.route('/login', methods=['POST', 'GET'])
def login():
    if f.request.method == 'POST':
        f.session['username'] = f.request.form['username']
        return f.redirect(f.url_for('index'))
    else:
        return f.render_template('login.html')


@app.route('/logout')
def logout():
    f.session.pop('username', None)
    return f.redirect(f.url_for('index'))


@app.route('/output', methods=['POST', 'GET'])
def tweet_search():
    query_tweet = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    if f.request.method == 'POST':
        search_word, date_since = ts.request_input()
        time_tweet_request = time.time()
        tweets = ts.get_tweet(search_word, date_since, api)
        time_tweet_fetch = time.time()
        print("Req-Fetch Tweet: ", time_tweet_fetch - time_tweet_request)
        time_loop_tweet = time.time()
        for i, tweet in enumerate(tweets):
            query_tweet.append(tweet.full_text)
            # print(str(i) + ":" + tweet.full_text)
        time_tweet_list = time.time()
        print("To list: ", time_tweet_list - time_loop_tweet)
        total_tweets = len(query_tweet)
        time_vector_load = time.time()
        vectorizer = joblib.load('vectorizer.sav')
        X = vectorizer.transform(query_tweet)
        time_model_load = time.time()
        svm = joblib.load('svm_classifier.sav')
        logr = joblib.load('logr_classifier.sav')
        time_done = time.time()
        print("Vector load: ", time_model_load - time_vector_load)
        print("Model load: ", time_done - time_model_load)
        prediction = 'True'
        t0 = time.time()
        svm_prediction = svm.predict(X)
        t1 = time.time()
        # print(type(svm_prediction))
        # svm_prediction = (svm_prediction)
        # print(type(svm_prediction))
        logr_prediction = logr.predict(X)
        t2 = time.time()
        print("Pred SVM: ", t1 - t0)
        print("Pred LR: ", t2 - t1)
        for pred in logr_prediction:
            if pred == 'positive':
                positive_count += 1
            elif pred == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
        precentage_positive = (positive_count / total_tweets) * 100
        precentage_negative = (negative_count / total_tweets) * 100
        precentage_neutral = (neutral_count / total_tweets) * 100

        t3 = time.time()
        fig = px.pie(names=['pos', 'neg', 'neut'],
                     values=[precentage_positive, precentage_negative, precentage_neutral], color=['r', 'b', 'g'])
        fig.show()
        print("Graph plot time: ", t3 - t2)
        print(precentage_positive, precentage_negative, precentage_neutral)
        # print(type(logr_prediction))
        # logr_prediction = (logr_prediction)
        # print(type(logr_prediction))
        # print(svm_prediction)
        # print(logr_prediction)
        template_values = {
            'query_tweet': query_tweet,
            'prediction': prediction,
            'logr_prediction': logr_prediction,
            'svm_prediction': svm_prediction
        }
        return f.render_template('index.html', **template_values)
    else:
        return f.render_template('login.html')


class TwitterStreamer:

    @staticmethod
    def authenticate():
        auth = tw.OAuthHandler(consumer_API_key, consumer_API_secret_key)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit=True)
        return api

    @staticmethod
    def get_tweet(search_words, date_since, api):
        tweets = tw.Cursor(
            api.search,
            q=search_words,
            lang="en",
            since=date_since,
            tweet_mode='extended'
        ).items(1000)
        return tweets

    @staticmethod
    def display_tweets(tweets):
        pd.set_option('display.max_columns', None)
        data = [[tweet.text, tweet.user.screen_name, tweet.user.location] for tweet in tweets]
        tweet_text = pd.DataFrame(data=data, columns=['tweets', 'user', "location"])
        print(tweet_text)
        return tweet_text

    @staticmethod
    def request_input():
        search_word = f.request.form['search word'] + '-filter:retweets'
        date_since = f.request.form['date']
        return search_word, date_since


if __name__ == '__main__':
    ts = TwitterStreamer()
    api = ts.authenticate()
    app.run(debug=True)
