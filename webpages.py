import requests
from bs4 import BeautifulSoup
#from lib2to3.btm_utils import tokens
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

url = 'https://medium.com/mind-cafe/you-will-destroy-yourself-financially-if-you-save-9d7ece62d05f'


class Question2:
    stop_words = set(stopwords.words("english"))

    def __init__(self, url):
        r = requests.get(url)
        # Extract HTML from Response object and print
        html = r.text
        # Create a BeautifulSoup object from the HTML
        soup = BeautifulSoup(html, "html5lib")
        # Get soup title
        soup.title
        # Get soup title as string
        soup.title.string
        # Get the text out of the soup and print it
        text = soup.get_text()

    def text(self):
        return self.text

    def tokenize(self, text):
        return word_tokenize(str(text))

    def to_lowercase(self, tokens):
        return [word.lower() for word in tokens]

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def fdistribution(self, tokenized_words):
        fdist = FreqDist(tokenized_words)
        fdist.plot(30, cumulative=False)
        plt.show()

    def tf_idf(self, document):
        pass

    def tf_idf_top10(self, document):
        pass


q2 = Question2(url)
text = q2.text()
tokenized_words = q2.tokenize(text)
lower_tokenized_words = q2.to_lowercase(tokenized_words)
without_stopwords = q2.remove_stopwords(lower_tokenized_words)
fdist = q2.fdistribution(without_stopwords)