"""
1) Self-reporting any hate speech
-- must be easily matchable
2) Stores this data
-- data must be easily accessible
3) Classifies whether the text is hate speech or
not using some pre defined model
 -- accesses the data from 2)'s data structure
"""

import numpy as np
import pandas as pd
import re
import tweepy
from tweepy import *
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import preprocessor as p
from textblob import TextBlob

""" Building a Model """


def preprocess(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    cleaned = BeautifulSoup(text, "lxml").text
    cleaned = cleaned.lower()
    cleaned = REPLACE_BY_SPACE_RE.sub(' ', cleaned)
    cleaned = BAD_SYMBOLS_RE.sub('', cleaned)
    cleaned = ' '.join(word for word in cleaned.split() if word not in STOPWORDS)
    return cleaned


# best model -> Doc2vec and Logistic Regression
# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
def build_model():
    return ""


""" Extracting Tweets """
# reference:
# https://towardsdatascience.com/extracting-twitter-data-pre-processing-and-sentiment-analysis-using-python-3-0-7192bd8b47cf
tweets = "data/twitter_data.csv"
COLS = ['id', 'created_at', 'source', 'original_text', 'clean_text', 'sentiment',
        'favorite_count', 'retweet_count', 'original_author', 'hashtags', 'user_mentions', 'location']

# specify certain
keywords = []

def write_tweets(keyword, file):
    for page in tweepy.Cursor(api.search, q=keyword,
                              count=200, include_rts=False).pages(10):
        for status in page:
            new_entry = []
            status = status._json

            ## check whether the tweet is in english or skip to the next tweet
            if status['lang'] != 'en':
                continue

            # when run the code, below code replaces the retweet amount and
            # no of favorires that are changed since last download.
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                        status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue

            # tweepy preprocessing called for basic preprocessing
            clean_text = p.clean(status['text'])

            # pass textBlob method for sentiment calculations
            blob = TextBlob(clean_text)
            Sentiment = blob.sentiment

            # new entry append
            new_entry += [status['id'], status['created_at'],
                          status['source'], status['text'], clean_text, Sentiment,
                          status['favorite_count'], status['retweet_count']]

            # to append original author of the tweet
            new_entry.append(status['user']['screen_name'])

            # hashtagas and mentiones are saved using comma separted
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)

            # get location of the tweet if possible
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)

            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)
            csvFile = open(file, 'a', encoding='utf-8')
    df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")

write_tweets(keywords, tweets)

""" Implementing the Model """


# takes in text and cleans it
def clean_text(text):
    space_pattern = '\s+'
    cleaned = re.sub(space_pattern, ' ', text)
    # cleaned = reg.sub(url_regex, 'URL', "")
    return cleaned


# checks the clean text in a classifier model
# model.predict(x) outputs:
# 0 for hateful
# 1 for not
# anything else if couldn't predict
def predict(x_val, model):
    prediction = model.predict(x_val)
    if prediction == 0:
        return "Hate Content"
    elif prediction == 1:
        return "Not Hate Content"
    else:
        return "Neutral/unidentifiable"


# takes in csv file
# possibly could be the spreadsheet of results from a google form?
# runs all the above functions accordingly
# appends the results to the original csv
def run_model(csv_name, model):
    input_csv = pd.read_csv(csv_name)
    text_to_classify = input_csv['Text']
    predictions = []
    for text in text_to_classify:
        cleaned = clean_text(text)
        result = predict(cleaned, model)
        predictions.append(result)

    input_csv['Results'] = predictions

# run_model(tweets, build_model())

"""
Didn't have time but could implement:
1) A classifier model - to predict if an input text is hate content or not
-- requires a testing and training dataset
2) An input system where users could type in hate words/phrases that 
might be relavent to that time's culture and add it to the classifier model 
-- easiest might be a google form linked to a webpage which can be easily converted into a csv file
3) A classifier model that could check how reliable a user is 
-- might ask for their background, qualifications, etc. and makes a prediction accordingly
4) 
"""
