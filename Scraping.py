# import os
import flask
import pandas as pd
import tweepy as tw

import Access_Keys

app = flask.Flask(__name__)
app.debug = True


@app.route('/')
def landingpage():
    return flask.render_template('index.html')


@app.route('/login')
def helloworld():
    print('Hello')
    # ts = TwitterStreamer()
    # api = ts.authenticate()
    search_words = input("Enter a topic that you are looking for") + " -filter:retweets"
    # date_since = input("Enter a date")
    # return search_words, date_since
    return search_words


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
                           since=date_since).items(50)
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
