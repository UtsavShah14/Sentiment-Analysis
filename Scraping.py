# import os
import flask as f
import pandas as pd
import tweepy as tw

import Access_Keys

app = f.Flask(__name__)
app.debug = True
app.secret_key = 'hello'


@app.route('/')
def index():
    # username = ""
    if 'username' in f.session:
        template_values = {
            'username': f.session['username']
        }
        return f.render_template('index.html', **template_values)
        # return username
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


@app.route('/process', methods=['POST', 'GET'])
def tweetSearch():
    if f.request.method == 'POST':
        search_word = f.request.form['search word']
        date_since = f.request.form['date']
        ts = TwitterStreamer()
        api = ts.authenticate()
        tweets = ts.get_tweet(search_word, date_since, api)
        # tweets_text = ts.display_tweets(tweets)
        for tweet in tweets:
            print(tweet.text)
        template_values = {
            # tweets_text
        }
        # for tweet in tweets:
        # #     print(tweet)
        return f.render_template('index.html', **template_values)
    else:
        return f.render_template('login.html')
    # print('Hello')
    # ts = TwitterStreamer()
    # api = ts.authenticate()
    # search_words = input("Enter a topic that you are looking for") + " -filter:retweets"
    # date_since = input("Enter a date")
    # return search_words, date_since


consumer_API_key = Access_Keys.consumer_API_key
consumer_API_secret_key = Access_Keys.consumer_API_secret_key
access_token = Access_Keys.access_token
access_token_secret = Access_Keys.access_token_secret


class TwitterStreamer:

    def __init__(self):
        api = self.authenticate()

    def authenticate(self):
        auth = tw.OAuthHandler(consumer_API_key, consumer_API_secret_key)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit=True)
        return api

    def get_tweet(self, search_words, date_since, api):
        tweets = tw.Cursor(api.search,
                           q=search_words,
                           lang="en",
                           since=date_since).items(5)
        return tweets

    def display_tweets(self, tweets):
        pd.set_option('display.max_columns', None)
        data = [[tweet.text, tweet.user.screen_name, tweet.user.location] for tweet in tweets]
        tweet_text = pd.DataFrame(data=data, columns=['tweets', 'user', "location"])
        print(tweet_text)
        return tweet_text

    def listener(self):
        pass

    def get_home_timeline(self):
        statuses = api.home_timeline()
        for i, status in enumerate(statuses):
            print(str(i) + ": " + status.text)

    def get_trends(self):
        trends = api.trends_place(2352824)
        for trend in trends[0]["trends"][:10]:
            print(trend["name"])


if __name__ == '__main__':
    # ts = TwitterStreamer()
    # api = ts.authenticate()
    # search_words = input("Enter a topic that you are looking for") + " -filter:retweets"
    # date_since = input("Enter a date")
    # tweets = ts.get_tweet(search_words, date_since, api)
    # tweet_text = ts.display_tweets(tweets)
    # tweet_text.to_csv("Tweets.csv", index=False, header=True)
    # ts.get_home_timeline()
    # trends = api.trends_available
    # for trend in trends:
    #     print(trend)
    app.run(debug=True)
