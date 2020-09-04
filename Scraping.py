import json

import pandas as pd
import textblob
import tweepy as tw

import Access_Keys
import preprocessing_data

data = {}  # Stores the gathered data from using tweepy and textblob


class TwitterStreamer:
    """
    The class is a collection of all the functions that will be used to scrape data from Twitter.
    Authentication of the API, preprocessing the data and getting the sentiment of the tweet defined with this class.
    """

    def __init__(self):
        # Keys for API authentication
        consumer_API_key = Access_Keys.consumer_API_key
        consumer_API_secret_key = Access_Keys.consumer_API_secret_key
        access_token = Access_Keys.access_token
        access_token_secret = Access_Keys.access_token_secret

        # Try to authenticate using the provided keys, if failed, it will raise an AuthenticationError
        try:
            self.auth = tw.OAuthHandler(consumer_API_key, consumer_API_secret_key)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tw.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        except tw.TweepError:
            print("Authentication Error: Check your Keys")

    @staticmethod
    # Using textblob, we get the sentiment of the tweet by a random user.
    # Sentiment is checked on cleaned text of the raw tweet
    # tweet_analysis stores a list of desired output
    def get_sentiment(status):
        analysis = textblob.TextBlob(preprocessing_data.get_clean_text(status))
        if analysis.sentiment.polarity > 0:
            tweet_analysis = [status, analysis.sentiment.polarity, analysis.sentiment.subjectivity, 'positive']
            return tweet_analysis
        elif analysis.sentiment.polarity < 0:
            tweet_analysis = [status, analysis.sentiment.polarity, analysis.sentiment.subjectivity, 'negative']
            return tweet_analysis
        else:
            tweet_analysis = [status, analysis.sentiment.polarity, analysis.sentiment.subjectivity, 'neutral']
            return tweet_analysis


class MyStreamListener(tw.StreamListener):
    """
    MyStreamListener class parameterizes tweepy's StreamListener class. This class overrides the default methods of
    tweepy's StreamListener class.
    MyStreamListener overrides on_data, on_status and on_error classes.
    """
    stream_stop = 0  # Control variable to stop listening to incoming tweets
    stream_limit = 2  # Counter variable to read N number of incoming tweets

    def on_data(self, raw_data):
        # Loads data into json format and parse it in a more accessible format (dictionary)
        json_data = json.loads(raw_data)
        status = tw.Status.parse(self.api, json_data)
        # We return false to stop the connection to Twitter when we reach the required tweets
        if self.stream_stop > self.stream_limit:
            return False
        else:
            self.stream_stop += 1
            self.on_status(status)
            return True

    def on_status(self, status):
        # Check if the status is retweeted, if True, we skip that tweet.
        try:
            if status.retweeted_status:
                return
        except AttributeError:
            # Check if the tweet(status) is a truncated text or fulltext.
            if not status.truncated:
                # We get the sentiment of the tweet.
                analysis = api.get_sentiment(status.text)
                # Try checking if the country exists in place, if not, put 'NA' in place
                try:
                    analysis.insert((len(analysis) - 1), status.place.country)
                except AttributeError:
                    analysis.insert((len(analysis) - 1), 'NA')
                data[status.id] = analysis
            else:
                analysis = api.get_sentiment(status.extended_tweet['full_text'])
                try:
                    analysis.insert((len(analysis) - 1), status.place.country)
                except AttributeError:
                    analysis.insert((len(analysis) - 1), 'NA')
                data[status.id] = analysis

    def on_error(self, status_code):
        # Used to stop connection to tweepy.
        if status_code == 420:
            return False


if __name__ == '__main__':
    # GeoBox code for three countries
    Ireland = -9.97708574059, 51.6693012559, -6.03298539878, 55.1316222195
    India = 68.1766451354, 7.96553477623, 97.4025614766, 35.4940095078
    United_States = -171.791110603, 18.91619, -66.96466, 71.3577635769

    # Creates an instance of the class. Used with myStream, instance of the StreamListener class
    api = TwitterStreamer()
    # Instance of StreamListener, passing above instance and MyStreamListener instance as parameters
    myStream = tw.Stream(auth=api.auth, listener=MyStreamListener(), tweet_mode='extended')
    print("Fetching tweets...")
    # Streams live tweeter data with location and language filters
    myStream.filter(locations=[-9.97708574059, 51.6693012559, -6.03298539878, 55.1316222195,
                               68.1766451354, 7.96553477623, 97.4025614766, 35.4940095078,
                               -171.791110603, 18.91619, -66.96466, 71.3577635769
                               ], languages=['en'])
    print("Generating your CSV...")
    # Adding the collected data from TwitterStreamer class
    dataframe_input = [
        [data[element][0], data[element][1], data[element][2], data[element][3], data[element][4]]
        for element in data]
    # The try catch block executes converting the data(list) to pandas and append the dataframe to a CSV data file.
    try:
        data_frame_result = pd.read_csv("Tweets.csv")
        data_frame = pd.DataFrame(
            data=dataframe_input,
            columns=['Text', 'Polarity', 'Subjectivity', 'Country', 'Sentiment'])
        try:
            data_frame.to_csv("Tweets.csv", encoding='UTF-8-sig', mode='a', index=False, header=False)
            print("Your CSV is ready!")
            print("Here's a look on your data...")
            c_s_v = pd.read_csv("Tweets.csv")
            print(c_s_v.head())  # Prints the head of the CSV file
            print(c_s_v.tail())  # Prints the tail of the CSV file
        except PermissionError:
            # Error if the file is open in the background in the time of writing to file
            print("An Error occurred\nPlease close the 'Tweets.csv' file and try again")
    except FileNotFoundError:
        # Error if the file is not found in the existing folder
        data_frame = pd.DataFrame(
            data=dataframe_input,
            columns=['Text', 'Polarity', 'Subjectivity', 'Country', 'Sentiment'])
        data_frame.to_csv("Tweets.csv", encoding='UTF-8-sig', mode='a', index=False, header=True)
        print("Your CSV is ready!")
        print("Here's a look on your data...")
        c_s_v = pd.read_csv("Tweets.csv")
        print(c_s_v.head())  # Prints the head of the CSV file
        print(c_s_v.tail())  # Prints the tail of the CSV file
