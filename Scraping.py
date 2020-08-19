# import os
import nltk
import pandas as pd
import tweepy as tw

import Access_Keys

# import re

# nltk.download('wordnet')
consumer_API_key = Access_Keys.consumer_API_key
consumer_API_secret_key = Access_Keys.consumer_API_secret_key
access_token = Access_Keys.access_token
access_token_secret = Access_Keys.access_token_secret


class TwitterStreamer:

    @staticmethod
    def authenticate():
        auth = tw.OAuthHandler(consumer_API_key, consumer_API_secret_key)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
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
        data = []
        for tweet in tweets:
            if tweet.lang == 'en':
                data.append([tweet.full_text, tweet.user.screen_name, tweet.user.location])
        tweet_text = pd.DataFrame(data=data, columns=['tweets', 'user', "location"])
        print(tweet_text)
        return tweet_text

    @staticmethod
    def get_home_timeline():
        statuses = api.home_timeline(tweet_mode='extended')
        for i, status in enumerate(statuses):
            if status.lang == 'en':
                print(str(i) + ": " + status.full_text)
        return statuses

    @staticmethod
    def get_trends():
        trends = api.trends_place(2352824)
        for trend in trends[0]["trends"][:10]:
            print(trend["name"])


class MyStreamListener(tw.StreamListener):

    def on_status(self, status):
        # with open('Tweets.csv', 'a', encoding='UTF-8') as file:
        # file.write(status.text)
        # file.write('\n')
        processed_tweet_text = preprocess_tweet_text(status.text)
        # no_stop_word_status = remove_stopwords(status.text)
        # lemmatized_status = lemmatize_status(no_stop_word_status)
        # print(type(status.text.split()))
        # print(type(no_stop_word))
        # print(status.text)
        # print(no_stop_word_status)
        # print(lemmatized_status)
        # file.write(status.text)
        print(processed_tweet_text)


basic_stopwords_list = {
    'a', 'an', 'all', 'and', 'are', 'as', 'at',
    'be', 'but', 'can', 'do', 'did', 'for',
    'get', 'give' 'has', 'had', 'have', 'how',
    'i', 'if', 'in', 'is', 'it',
    'me', 'my', 'no',
    'of', 'on', 'or',
    'that', 'the', 'there' 'this', 'to', 'up',
    'was', 'we', 'what', 'when', 'why', 'where', 'would', 'with', 'will',
    'you'
}


def preprocess_tweet_text(corpus):
    return lemmatize_status(remove_stopwords(corpus))


def remove_stopwords(status_text):
    symbols = '!"#$%&\'()*+,-./"?:;<=>[\\]^_`{|}~'
    stopword_removed_status_text = []
    for word in status_text.lower().split():
        if word.strip(symbols) not in basic_stopwords_list:
            if word.strip(symbols) != '' or ' ':
                stopword_removed_status_text.append(word.strip(symbols))
    full_string = ' '.join(stopword_removed_status_text)
    # word = word.strip(symbols)
    # if word not in basic_stopwords_list:
    #     if word != ' ':
    #         stopword_removed_status_text.append(''.join(word))
    # for status in status_text:
    #     # status = re.sub('\W+', ' ', status)
    #     stopword_removed_status_text.append(
    #         ' '.join([word.strip(symbols) for word in status.split()
    #                   if word not in basic_stopwords_list]))
    return full_string


def lemmatize_status(status):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_status = []
    for word in status.split():
        lemmatized_status.append(lemmatizer.lemmatize(word))
    full_string = ' '.join(lemmatized_status)
    return full_string


if __name__ == '__main__':
    Ireland = -9.97708574059, 51.6693012559, -6.03298539878, 55.1316222195
    ts = TwitterStreamer()
    api = ts.authenticate()
    myStreamListener = MyStreamListener()
    myStream = tw.Stream(auth=api.auth, listener=myStreamListener)
    myStream.filter(locations=[68.1766451354, 7.96553477623, 97.4025614766, 35.4940095078], languages=['en'])
    # search_words = input("Enter a topic that you are looking for") + " -filter:retweets"
    # date_since = input("Enter a date")
    # tweets = ts.get_tweet(search_words, date_since, api)
    # tweet_text = ts.display_tweets(tweets)
    # tweet_text.to_csv("Tweets.csv", index=False, header=True)
    # ts.get_home_timeline()
    # tweets = api.home_timeline(languages=['pt'])
    # for i, tweet in enumerate(tweets):
    #     print(str(i) + ": " + tweet.text)
    # trends = api.trends_available
    # for trend in trends:
    #     print(trend)
    # tweets = tw.Cursor(api.home_timeline,
    #                    tweet_mode='extended',
    #                    language=['en']).items(5)
    # for i, tweet in enumerate(tweets):
    #     print(str(i) + ": " + tweet.full_text)
    # status = api.get_status(1, tweet_mode="extended")
    # try:
    #     print(status.retweeted_status.full_text)
    # except AttributeError:  # Not a Retweet
    #     print(status.full_text)
    # tweets = ts.get_home_timeline()
    # tweet_text = ts.display_tweets(tweets)
    # tweet_text.to_csv("Tweets.csv", index=False, header=True)
    # statuses = api.home_timeline(tweet_mode='extended')
    # status_text = []
    # for status in tweets:
    #     status_text.append(status.full_text)
    # print("Before stopword removal\n")
    # print([status for status in status_text])
    # stopword_removed_status_text = stopword_removal(status_text)
    # print("After stopword removal")
    # for status in status_text:
    #     stopword_removed_status_text = []
    #     stopword_removed_status_text.append(
    #         ' '.join([word for word in status.split()
    #                   if word not in english_stopwords_list]))
    # print(temp)
    # print([status for status in stopword_removed_status_text])
    # print(stopwords)
    # lematizer = nltk.stem.WordNetLemmatizer()
    # print(lematizer.lemmatize('saw'))
