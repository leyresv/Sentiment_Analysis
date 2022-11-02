import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from data.preprocess_tweets import get_twitter_dataset


def train_tweet_classifier(X_train, Y_train, X_test, Y_test):
    """
    Train different tweets classifier and save the best performing one

    :param X_train: train input features
    :param Y_train: train labels
    :param X_test: test input features
    :param Y_test: test labels
    """
    # Train different tweets classifiers and keep the best one

    # Logistic Regression classifier
    lr_classifier = LogisticRegression(random_state=42)
    lr_classifier.fit(X_train, Y_train)

    # SVM classifier
    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(X_train, Y_train)

    # Naïve Bayes classifier
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, Y_train)

    classifiers = [lr_classifier, svm_classifier, nb_classifier]
    accuracies = []
    for classifier in classifiers:
        Y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(Y_test, Y_pred))
    best_classifier = classifiers[np.argmax(accuracies)]

    filename = "tweet_classifier.sav"
    pickle.dump(best_classifier, open(filename, "wb"))


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = get_twitter_dataset()
    train_tweet_classifier(X_train, Y_train, X_test, Y_test)