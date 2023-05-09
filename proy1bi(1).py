

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Instalación de librerias
import pandas as pd
import numpy as np
import sys

import re, string, unicodedata
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer,  TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score


import matplotlib.pyplot as plt



def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
      new_words.append(word.lower())
    return new_words
      
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stops = set(stopwords.words('spanish'))
    new_words= []
    for word in words:
      if word not in stops:
        new_words.append(word) 
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    return [stemmer.stem(token) for token in words]

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token, pos='v') for token in words]

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems + lemmas


def preprocessing(words):
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words


from sklearn.preprocessing import FunctionTransformer

import joblib


data_p=pd.read_csv('HotelsReviews.csv', sep=',',index_col=0, encoding = 'utf-8')
# Train the MaxEnt classifier
pipeline = Pipeline([
    ('preprocess', FunctionTransformer(preprocessing)),
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])
textos = data_p.copy()
pipeline.fit(textos['review_text'], textos['label'])

results = pipeline.predict(textos['review_text'])

print(results)

joblib.dump(pipeline, 'trained_model.joblib')



from flask import Flask, request, jsonify

app = Flask(__name__)

import joblib

model = joblib.load('trained_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    datajson = request.get_json() # assumes JSON data
    data = pd.json_normalize(datajson)
    # make predictions with your model
    predictions = model.predict(data['review_text'])
    print(data)
    # return predictions as JSON
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
