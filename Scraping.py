# import os
import pandas as pd
import tweepy as tw

import Access_Keys

start_url = ""
consumer_API_key = Access_Keys.consumer_API_key
consumer_API_secret_key = Access_Keys.consumer_API_secret_key
access_token = Access_Keys.access_token
access_token_secret = Access_Keys.access_token_secret

auth = tw.OAuthHandler(consumer_API_key, consumer_API_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# api.update_status("Look, I'm tweeting from #Python in my #BigData Thesis!")

search_words = "#wildfires"
date_since = "2018-11-16"
#
# tweets = tw.Cursor(api.search,
#               q=search_words,
#               lang="en",
#               since=date_since).items(5)
# tweets
#
tweets = tw.Cursor(api.search,
                   q=search_words,
                   lang="en",
                   since=date_since).items(5)

# print([tweet.text for tweet in tweets])

users_locs = [[tweet.text, tweet.user.screen_name, tweet.user.location] for tweet in tweets]
# users_locs

pd.set_option('display.max_columns', None)
tweet_text = pd.DataFrame(data=users_locs,
                          columns=['tweets', 'user', "location"])
print(tweet_text)

tweet_text.to_csv("Tweets.csv", index=False, header=True)
