import flask as f
import joblib
import tweepy as tw
import Access_Keys
from country import country_bounding_boxes
import preprocessing_data

# Access Keys loading
consumer_API_key = Access_Keys.consumer_API_key
consumer_API_secret_key = Access_Keys.consumer_API_secret_key
access_token = Access_Keys.access_token
access_token_secret = Access_Keys.access_token_secret

app = f.Flask(__name__)  # Initializing the flask instance
app.secret_key = 'AnySecretKeyHere'  # Used for protection from potential attacks


# This is where our application starts. Its the landing page for our web application
@app.route('/')
def index():
    # Template values to render for the web template
    template_values = {
        'country_list': country_bounding_boxes,
    }
    return f.render_template('index.html', **template_values)


# Accepts the post request of user input form submission
# This function does the main logical part of our web app
# From fetching data from twitter to processing and classifying it to the target class
# and processing the result to display a graphical representation of the result is done here
@app.route('/output', methods=['POST'])
def tweet_search():
    if f.request.method == 'POST':
        search_word, date_since, location = ts.request_input()  # Collect user inputs
        tweets = ts.get_tweet(search_word, date_since, location, api)  # Request twitter for data
        query_tweet, query_location = ts.tweet_to_list(tweets)  # Convert twitter response to list for ease of access

        # If the retrieved result from twitter is None (empty), flashes aan error message
        if not query_tweet:
            f.flash("Could not retrieve any tweets")
            return f.redirect(f.url_for('index'))

        # Request our selected model to predict sentiments of data retrieved from twitter
        prediction = ts.predict_values(query_tweet)

        # Total number of tweets that we have
        total_tweets = len(query_tweet)

        # Get the percentage for each label of the target class
        percentage_positive, percentage_negative, percentage_neutral = ts.get_percentage(
            prediction, total_tweets)

        # Graphical representation of retrieved data
        graph_data = ts.plot_graph(percentage_positive, percentage_negative, percentage_neutral)

        template_values = {
            'country_list': country_bounding_boxes,
            'graph_data': graph_data,
            'query_tweet': query_tweet,
            'prediction': True,
        }
        return f.render_template('index.html', **template_values)
    else:
        return f.render_template('login.html')


class TwitterStreamer:
    """
    This class is a collection of all the function that will be used fetch data from twitter as well as process it in
    order to get our final output.
    The class has all the functions required for optimal functioning of the web application.
    """

    @staticmethod
    # Authenticate our credentials tweepy credentials
    def authenticate():
        auth = tw.OAuthHandler(consumer_API_key, consumer_API_secret_key)
        auth.set_access_token(access_token, access_token_secret)
        api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        return api

    @staticmethod
    # Search twitter with the search API to fetch data as per user query
    def get_tweet(search_words, date_since, location, api_call):
        max_tweets = 500  # Limit on number of tweets fetched
        # For testing purpose, keep the count low because Tweepy has a rate limit which will halt the execution
        tweets = tw.Cursor(
            api_call.search,
            q=search_words,
            geocode=location,
            lang="en",
            since=date_since,
            tweet_mode='extended'
        ).items(max_tweets)
        return tweets

    @staticmethod
    # Get the user input and process it in required format
    def request_input():
        search_word = f.request.form['search word'] + '-filter:retweets'  # Ignoring the retweets here
        date_since = f.request.form['date']
        geocode = f.request.form['location']  # Required in geobox format (lat, long, radius)
        if geocode != 'world':
            geocode = list(country_bounding_boxes[geocode][1])
            geocode.pop()
            geocode.pop()
            geocode.append('3000km')  # Setting the radius
            location = ','.join([str(elem) for elem in geocode])  # Converted into required format
        else:
            location = None
        return search_word, date_since, location

    @staticmethod
    # Preprocess tweets and convert it to list for ease of use
    def tweet_to_list(tweets):
        query_tweet = []
        query_location = []
        for tweet in tweets:
            query_tweet.append(preprocessing_data.get_clean_text(tweet.full_text))
            query_location.append(tweet.user.location)
        return query_tweet, query_location

    @staticmethod
    # Load the pickled model and predict category for fetched tweets
    def predict_values(query_tweet):
        vectorizer = joblib.load('vectorizer.sav')  # Loading vector
        X = vectorizer.transform(query_tweet)  # Vectorizing
        model = joblib.load('final_model.sav')  # Loading our model
        prediction = model.predict(X)  # Predicting the category
        return prediction

    @staticmethod
    # Get percentage of each class in the target attribute
    def get_percentage(prediction, total_tweets):
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        for pred in prediction:
            if pred == 'positive':
                positive_count += 1
            elif pred == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
        percentage_positive = (positive_count / total_tweets) * 100
        percentage_negative = (negative_count / total_tweets) * 100
        percentage_neutral = (neutral_count / total_tweets) * 100
        return percentage_positive, percentage_negative, percentage_neutral

    @staticmethod
    # using matplotlib.plotly to plot a doughnut chart for our collected data
    def plot_graph(percentage_positive, percentage_negative, percentage_neutral):
        categories = ['positive', 'negative', 'neutral']  # Categories of our target attribute
        values = [percentage_positive, percentage_negative, percentage_neutral]  # Input values for the graph
        color = ['#158467', '#e84a5f', '#fddb3a']  # Color for each category
        data = [{
            'values': values,
            'labels': categories,
            'hole': '0.4',  # To make it a doughnut
            'type': 'pie',  # Type of graph
            'marker': {
                'colors': color
            },
            'textinfo': 'label+percent',
            'hoverinfo': 'label+percent',
        }]
        return data


if __name__ == '__main__':
    ts = TwitterStreamer()
    api = ts.authenticate()
    app.run(debug=True)
