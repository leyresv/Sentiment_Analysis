import pickle
from data.preprocess_tweets import extract_freq_feature

DOMAINS = ["tweet"]


class SentimentClassifier:
    """
    Sentiment analyzer
    """

    def __init__(self, domain):
        if domain not in DOMAINS:
            raise ValueError(f"Domain not recognized. Choose between: {', '.join(DOMAINS)}")
        self.domain = domain
        self.classifier = pickle.load(open(f"../models/{self.domain}_classifier.sav", "rb"))

    def predict_class(self, text):
        """
        Predict sentiment of the input text

        :param text: input text (string)
        :return: sentiment label
        """
        if self.domain == "tweet":
            vocab_dict = pickle.load(open("../data/tweets_vocab.pkl", "rb"))
            processed_text = extract_freq_feature([text], vocab_dict)
            pred = self.classifier.predict(processed_text)
            if pred == 1:
                return "Positive"
            else:
                return "Negative"


if __name__ == "__main__":
    domain = input("What kind of data do you want to analyze? Choose between: tweet, sentence, review ")
    classifier = SentimentClassifier(domain)
    text = input(f"Enter your {domain} (type 'close' to exit): ")
    while text != "close":
        prediction = classifier.predict_class(text)
        print(f"Sentiment: {prediction}")
        text = input(f"Enter your {domain} (type 'close' to exit): ")

