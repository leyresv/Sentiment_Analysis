# Sentiment Analysis on different types of texts

In this project, I have implemented a sentiment analyzer that is able to classify different types of input text (tweets, movie reviews, sentences).


## Installation
```bash
pip install git+https://github.com/leyresv/Sentiment_Analysis.git
pip install -r requirements.txt
```

## Usage



## Models

Here you can find the different models and the code to train them:

### Tweets classifier

The tweets classifier has been trained using the *Twitter Samples* dataset from NLTK. 
I have extracted the positive and negative frequencies of the words on each tweet and used them as features to train three different classification models using
a Logistic Regression, a Naïve Bayes and a Support Vector Machine algorithms. I only keep the best performing one, which is the Logistic Regression classifier,
with an accuracy of 0.997 on the test set.

## Visualization
