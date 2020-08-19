import json

import nltk
import textblob
import tweepy as tw

import Access_Keys

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

data = {}


class TwitterStreamer:

    def __init__(self):
        consumer_API_key = Access_Keys.consumer_API_key
        consumer_API_secret_key = Access_Keys.consumer_API_secret_key
        access_token = Access_Keys.access_token
        access_token_secret = Access_Keys.access_token_secret
        try:
            self.auth = tw.OAuthHandler(consumer_API_key, consumer_API_secret_key)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tw.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        except tw.TweepError:
            print("Authentication Error: Check your Keys")

    def preprocess_tweet_text(self, corpus):
        return self.lemmatize_status(self.remove_stopwords(corpus))

    @staticmethod
    def remove_stopwords(status_text):
        symbols = '!"#$%&\'()*+,-./"?:;<=>[\\]^_`{|}~'
        stopword_removed_status_text = []
        for word in status_text.lower().split():
            if word.strip(symbols) not in basic_stopwords_list:
                if word.strip(symbols) != '' or ' ':
                    stopword_removed_status_text.append(word.strip(symbols))
        full_string = ' '.join(stopword_removed_status_text)
        return full_string

    @staticmethod
    def lemmatize_status(status):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        lemmatized_status = []
        for word in status.split():
            lemmatized_status.append(lemmatizer.lemmatize(word))
        full_string = ' '.join(lemmatized_status)
        return full_string

    @staticmethod
    def get_sentiment(status):
        analysis = textblob.TextBlob(api.preprocess_tweet_text(status))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'


class MyStreamListener(tw.StreamListener):
    stream_stop = 0
    stream_limit = 3

    def on_data(self, raw_data):
        json_data = json.loads(raw_data)
        status = tw.Status.parse(self.api, json_data)
        print(self.stream_stop)
        if self.stream_stop > self.stream_limit:
            return False
        else:
            self.stream_stop += 1
            self.on_status(status)
            return True

    def on_status(self, status):
        print("in myStream")
        try:
            if status.retweeted_status:
                return
        except AttributeError:
            if not status.truncated:
                analysis = api.get_sentiment(status.text)
                # print(status.text + ": " + analysis)
                data.update({status.text: analysis})
                # print(data)
            else:
                analysis = api.get_sentiment(status.extended_tweet['full_text'])
                # print(status.extended_tweet['full_text'] + ": " + analysis)
                data.update({status.extended_tweet['full_text']: analysis})
                # print(data)

    def on_error(self, status_code):
        if status_code == 420:
            return False


if __name__ == '__main__':
    Ireland = -9.97708574059, 51.6693012559, -6.03298539878, 55.1316222195
    India = 68.1766451354, 7.96553477623, 97.4025614766, 35.4940095078
    United_States = -171.791110603, 18.91619, -66.96466, 71.3577635769

    api = TwitterStreamer()
    myStream = tw.Stream(auth=api.auth, listener=MyStreamListener(), tweet_mode='extended')
    # print("called myStream")
    myStream.filter(locations=[-9.97708574059, 51.6693012559, -6.03298539878, 55.1316222195], languages=['en'])
    # print("updated the value of corpus")
    print(data)
