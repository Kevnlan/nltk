import math
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import requests
#from lib2to3.btm_utils import tokens
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import regex as re

url1 = 'https://www.gutenberg.org/files/2701/2701-h/2701-h.htm' #Moby Dick
url2 = 'https://en.wikipedia.org/wiki/Atomic_bombings_of_Hiroshima_and_Nagasaki' #Attomic bombings of Hiroshima and Nagasaki
url3 = 'https://en.wikipedia.org/wiki/Archduke_Franz_Ferdinand_of_Austria' #History of Archduke Franz Ferdinand
url4 = 'https://en.wikipedia.org/wiki/Batman' #Batman
url5 = 'https://www.gutenberg.org/files/730/730-h/730-h.htm' #Oliver Twist
url6 = 'https://www.icanw.org/hiroshima_and_nagasaki_bombings#:~:text=The%20uranium%20bomb%20detonated%20over,chronic%20disease%20among%20the%20survivors.' #Impact of Atomic Bombings on Hiroshima and Nagasaki

class Question2:
    stop_words = set(stopwords.words("english"))

    def __init__(self, url):
        self.r = requests.get(url)
        # Extract HTML from Response object and print
        self.html = self.r.text
        # Create a BeautifulSoup object from the HTML
        self.soup = BeautifulSoup(self.html, "html5lib")
        # Get soup title
        self.soup.title
        # Get soup title as string
        self.soup.title.string
        # Get the text out of the soup and print it
        self.text_string = self.soup.get_text()

    def text(self):
        return self.text_string

    def tokenize(self, text):
        return word_tokenize(text)

    def to_lowercase(self, tokens):
        return [word.lower() for word in tokens]

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def fdistribution(self, tokenized_words):
        fdist = FreqDist(tokenized_words)
        fdist.plot(30, cumulative=False)
        plt.show()

    def tf(self, without_sws):
        tf_dict_local = {}

        # Remove punctuation from the input string
        bag_of_words = [word for word in without_sws if re.match("^\P{P}(?<!-)", word)]

        document_size = len(bag_of_words)

        # Get unique words from the string without punctuation
        unique_words = set(bag_of_words)

        # Create dictionary with number of occurrences of words in the corpus
        word_count_dict = dict.fromkeys(unique_words, 0)

        # Iterate through bag of words and record the number of times a word appears
        for word in bag_of_words:
            word_count_dict[word] += 1

        # Calculate the tf of each word and add it to a dictionary
        for word, count in word_count_dict.items():
            tf_dict_local[word] = count / float(document_size)

        return tf_dict_local

    def idf(self, documents):
        document_size = len(documents)
        idf_dict = dict.fromkeys(documents[0].keys(), 0)
        for document in documents:
            for word, value in document.items():
                if value > 0:
                    idf_dict[word] = math.log(document_size / float(value))
        return idf_dict

    def tf_idf(self, tf_dict, idf):
        tf_idf_dict_local = {}
        for word, value in tf_dict.items():
            tf_idf_dict_local[word] = tf_dict[word] * idf[word]
        return tf_idf_dict_local

    def tf_idf_top10(self, document):
        # Sort the input dictionary and output as a list
        sorted_dict = sorted(document.items(), key=lambda x: x[1], reverse=True)

        # Trim the list and convert it back to a dictionary
        first10vals = dict(sorted_dict[:10])
        return first10vals

q1 = Question2(url1)
text = q1.text()
tokenized_words = q1.tokenize(text)
lower_tokenized_words  = q1.to_lowercase(tokenized_words)
without_stopwords = q1.remove_stopwords(lower_tokenized_words)
fdist = q1.fdistribution(without_stopwords)

q2 = Question2(url2)
text = q2.text()
tokenized_words = q2.tokenize(text)
lower_tokenized_words  = q2.to_lowercase(tokenized_words)
without_stopwords = q2.remove_stopwords(lower_tokenized_words)
fdist = q2.fdistribution(without_stopwords)

q3 = Question2(url3)
text = q3.text()
tokenized_words = q3.tokenize(text)
lower_tokenized_words  = q3.to_lowercase(tokenized_words)
without_stopwords = q3.remove_stopwords(lower_tokenized_words)
fdist = q3.fdistribution(without_stopwords)

q4 = Question2(url4)
text = q4.text()
tokenized_words = q4.tokenize(text)
lower_tokenized_words  = q4.to_lowercase(tokenized_words)
without_stopwords = q4.remove_stopwords(lower_tokenized_words)
fdist = q4.fdistribution(without_stopwords)

q5 = Question2(url5)
text = q5.text()
tokenized_words = q5.tokenize(text)
lower_tokenized_words  = q5.to_lowercase(tokenized_words)
without_stopwords = q5.remove_stopwords(lower_tokenized_words)
fdist = q5.fdistribution(without_stopwords)

q6 = Question2(url6)
text = q6.text()
tokenized_words = q6.tokenize(text)
lower_tokenized_words  = q6.to_lowercase(tokenized_words)
without_stopwords = q6.remove_stopwords(lower_tokenized_words)
fdist = q6.fdistribution(without_stopwords)


# Download web page from the internet
website_1 = Question2(url1)
website_2 = Question2(url2)
website_3 = Question2(url3)
website_4 = Question2(url4)
website_5 = Question2(url5)
website_6 = Question2(url6)

# Strip html elements and get the text from each of the documents
text_website_1 = website_1.text()
text_website_2 = website_2.text()
text_website_3 = website_3.text()
text_website_4 = website_4.text()
text_website_5 = website_5.text()
text_website_6 = website_6.text()

# Tokenize each of the documents
tokenized_web_1 = website_1.tokenize(text_website_1)
tokenized_web_2 = website_2.tokenize(text_website_2)
tokenized_web_3 = website_3.tokenize(text_website_3)
tokenized_web_4 = website_4.tokenize(text_website_4)
tokenized_web_5 = website_5.tokenize(text_website_5)
tokenized_web_6 = website_5.tokenize(text_website_6)

# Convert each of the tokenized documents to lowercase
lowercase_web_1 = website_1.to_lowercase(tokenized_web_1)
lowercase_web_2 = website_2.to_lowercase(tokenized_web_2)
lowercase_web_3 = website_3.to_lowercase(tokenized_web_3)
lowercase_web_4 = website_4.to_lowercase(tokenized_web_4)
lowercase_web_5 = website_5.to_lowercase(tokenized_web_5)
lowercase_web_6 = website_5.to_lowercase(tokenized_web_6)

# Remove all stopwords from each of the documents
stopwords_web_1 = website_1.remove_stopwords(lowercase_web_1)
stopwords_web_2 = website_2.remove_stopwords(lowercase_web_2)
stopwords_web_3 = website_3.remove_stopwords(lowercase_web_3)
stopwords_web_4 = website_4.remove_stopwords(lowercase_web_4)
stopwords_web_5 = website_5.remove_stopwords(lowercase_web_5)
stopwords_web_6 = website_5.remove_stopwords(lowercase_web_6)

# Evaluate the tf of each document
tf_doc_1 = website_1.tf(stopwords_web_1)
tf_doc_2 = website_2.tf(stopwords_web_2)
tf_doc_3 = website_3.tf(stopwords_web_3)
tf_doc_4 = website_4.tf(stopwords_web_4)
tf_doc_5 = website_5.tf(stopwords_web_5)
tf_doc_6 = website_6.tf(stopwords_web_6)

# Evaluate the idf of all documents
all_documents_idf = website_1.idf([tf_doc_1, tf_doc_2, tf_doc_3, tf_doc_4, tf_doc_5, tf_doc_6])

# Evaluate the tf-idf of each document
tf_idf_1 = website_1.tf_idf(tf_doc_1, all_documents_idf)
tf_idf_2 = website_2.tf_idf(tf_doc_2, all_documents_idf)
tf_idf_3 = website_3.tf_idf(tf_doc_3, all_documents_idf)
tf_idf_4 = website_4.tf_idf(tf_doc_4, all_documents_idf)
tf_idf_5 = website_5.tf_idf(tf_doc_5, all_documents_idf)
tf_idf_6 = website_6.tf_idf(tf_doc_6, all_documents_idf)

# Output the top 10 elements with the highest tf_idfs
print(website_1.tf_idf_top10(tf_idf_1))
print(website_2.tf_idf_top10(tf_idf_2))
print(website_3.tf_idf_top10(tf_idf_3))
print(website_4.tf_idf_top10(tf_idf_4))
print(website_5.tf_idf_top10(tf_idf_5))
print(website_6.tf_idf_top10(tf_idf_6))