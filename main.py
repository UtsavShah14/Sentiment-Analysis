# import os
import flask as f
import pandas as pd
import tweepy as tw

import Access_Keys

consumer_API_key = Access_Keys.consumer_API_key
consumer_API_secret_key = Access_Keys.consumer_API_secret_key
access_token = Access_Keys.access_token
access_token_secret = Access_Keys.access_token_secret

app = f.Flask(__name__)
app.secret_key = 'hello'


@app.route('/')
def index():
    # username = ""
    if 'username' in f.session:
        template_values = {
            'username': f.session['username']
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
    if f.request.method == 'POST':
        search_word, date_since = ts.request_input()
        tweets = ts.get_tweet(search_word, date_since, api)
        for i, tweet in enumerate(tweets):
            print(str(i) + ":" + tweet.text)
        template_values = {
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
        tweets = tw.Cursor(api.search,
                           q=search_words,
                           lang="en",
                           since=date_since).items(5)
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
        search_word = f.request.form['search word']
        date_since = f.request.form['date']
        return search_word, date_since


if __name__ == '__main__':
    ts = TwitterStreamer()
    api = ts.authenticate()
    app.run(debug=True)