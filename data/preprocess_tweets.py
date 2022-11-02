import nltk
import re
import string
import pickle
import numpy as np

from nltk.corpus import twitter_samples, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

# Download nltk resources
nltk.download("twitter_samples", quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download("stopwords", quiet=True)


def process_tweet(tweet):
    """
    Clean and tokenize a tweet

    :param tweet: string
    :return: list of tokens
    """
    # Regex cleaning
    tweet2 = re.sub(r"https?:\/\/\S*", "", tweet)   # Remove hashtags and hyperlinks
    tweet3 = re.sub(r"#", "", tweet2)               # Remove hash signs

    # Tweets tokenization
    tokenizer = TweetTokenizer(preserve_case=False,
                                strip_handles=True,     # Remove username handles
                                reduce_len=True)        # Replace repeated character sequences of length 3 or greater with sequences of length 3
    tweet_tokens = tokenizer.tokenize(tweet3)

    # Remove stop words and punctuation, and lemmatize
    stopwords_eng = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tweet_clean = [lemmatizer.lemmatize(token) for token in tweet_tokens if token not in stopwords_eng and token not in string.punctuation]

    return tweet_clean


def create_word_freqs_dict(tweets, labels):
    """
    Create frequencies dictionary

    :param tweets: list of tweets
    :param labels: list of labels (0 or 1)
    :return: vocabulary frequencies dictionary
    """
    tweets = [process_tweet(tweet) for tweet in tweets]
    word_freqs = {}
    for tweet, label in zip(tweets, labels):
        for word in tweet:
            if not (word, label) in word_freqs:
                word_freqs[(word, label)] = 0
            word_freqs[(word, label)] += 1
    # return dictionary sorted by values
    return dict(sorted(word_freqs.items(), key=lambda x:x[1], reverse=True))


def extract_freq_feature(tweets, vocab):
    """
    Convert tweets to frequency vectors

    :param tweets: list of tweets
    :param vocab: frequencies dictionary. key: tuple of (token string, sentiment int). value: frequency (int) of the token in the training data
    :return: numpy array of frequency feature (shape: number of tweets x 3)
    """

    freq_feature = []
    for tweet in tweets:
        tweet = process_tweet(tweet)
        pos = 0
        neg = 0
        # Ignore repeated words
        for word in list(set(tweet)):
            pos += vocab.get((word, 1), 0)
            neg += vocab.get((word, 0), 0)
        # Add 1 for bias
        freq_feature.append([1, pos, neg])
    return np.array(freq_feature)


def get_twitter_dataset():
    """
    Import twitter dataset and preprocess data

    :return: train and test features and labels
    """
    #Import dataset
    pos_tweets = twitter_samples.strings("positive_tweets.json")
    neg_tweets = twitter_samples.strings("negative_tweets.json")

    # Split train/test datasets
    train_pos, train_neg = pos_tweets[:4000], neg_tweets[:4000]
    test_pos, test_neg = pos_tweets[4000:], neg_tweets[4000:]
    train_tweets = train_pos + train_neg
    test_tweets = test_pos + test_neg

    # Create sentiment labels
    Y_train = np.append(np.ones((len(train_pos))), np.zeros((len(train_neg))))
    Y_test = np.append(np.ones((len(test_pos))), np.zeros((len(test_neg))))

    # Create frequencies dictionary
    vocab_dict = create_word_freqs_dict(train_tweets, Y_train)
    pickle.dump(vocab_dict, open("../data/tweets_vocab.pkl", "wb"))

    # Extract frequency features (1, pos_value, neg_value) for each tweet
    X_train = extract_freq_feature(train_tweets, vocab_dict)
    X_test = extract_freq_feature(test_tweets, vocab_dict)

    return X_train, Y_train, X_test, Y_test
