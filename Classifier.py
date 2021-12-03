import pandas as pd
import numpy as np
import math
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import WordNetLemmatizer
from nltk import pos_tag

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


df = pd.read_csv('car-reviews.csv')

def negations(data):
    data=data.replace("n't", " not")
    data=data.replace("n’t", " not")
    return data

def shortenings(data):
    data = data.replace("'ve", " have")
    data = data.replace("'re", " are")
    data = data.replace("'d", " would")
    return data

def tokenize(data):
    data = word_tokenize(data)
    return data

lmtzr = WordNetLemmatizer()
lem_code = {"NN" : "n", "JJ" : "a", "VB" : "v", "RB" : "r"}
def lemmatize(data):
    tags = pos_tag(data)
    lemmas = [lmtzr.lemmatize(word, lem_code[tag[:2]])
              for word, tag in tags if tag[:2] in lem_code.keys()]
    return lemmas


def lowercase(data):
    return [word.lower() for word in data]

stop_words = set(stopwords.words('english'))
def stopwords(data):
    data = [word for word in data if not word in stop_words or word in ['no','nor','not']]
    return data

def preprocessing(data):
    data = negations(data)
    data = shortenings(data)
    data = tokenize(data)
    data = lemmatize(data)
    data = lowercase(data)
    data = stopwords(data)
    return data

print(df.head())

#preprocessing review column, making sentiment column numerical
df['Review']=df['Review'].apply(lambda x: preprocessing(x))
df['Sentiment']=df['Sentiment'].map({'Pos':1,'Neg':0})

#after preprocessing
print(df.head())

# Sample positives and negatives separately in order for the train and test data to have the same label proportions
pos_index = [i for i in df.index if df.Sentiment.loc[i] == 1]
neg_index = [i for i in df.index if df.Sentiment.loc[i] == 0]

#create new lists with 80% training data and 20% testing data
#for both positive and negative data
train_pos = pos_index[:int(0.8 * len(pos_index))]
test_pos = pos_index[int(0.8 * len(pos_index)):]
train_neg = neg_index[:int(0.8 * len(neg_index))]
test_neg = neg_index[int(0.8 * len(neg_index)):]

#concatenate the negative and positive data.
X_train = df.loc[train_pos + train_neg, "Review"]
X_test = df.loc[test_pos + test_neg, "Review"]
y_train = df.loc[train_pos + train_neg, "Sentiment"]
y_test = df.loc[test_pos + test_neg, "Sentiment"]

#Preprocessed training data
X_train

# Feature selection to find the most common words
appearance_dict = dict()

for i in range(X_train.shape[0]):
    visited = set()
    for word in X_train.iloc[i]:
        if word not in visited:
            try:
                appearance_dict[word] += 1
                visited.add(word)
            except KeyError:
                appearance_dict[word] = 1
                visited.add(word)


cdf = pd.DataFrame([appearance_dict], index=["Count"]).T

#Choosing 20 as the threshold for reviews as it results in approx 1000 rows.
#Lowering threshold doesnt increase accuracy, but does increase processing time.

feature_list = cdf[cdf.Count > 20].index.to_list()

#creating count vectorizer with the feature list
count_vectorizer = CountVectorizer(vocabulary=feature_list)

#creating vector for number of time stem words appear in the training data
countvector_train = count_vectorizer.fit_transform(X_train.apply(lambda x : " ".join(x)))
countvector = pd.DataFrame(data = countvector_train.toarray(),columns = count_vectorizer.get_feature_names())
print(countvector)

#Implement multinomial Naive Bayes
MNB = MultinomialNB()
MNB.fit(countvector_train,y_train)

countvector_test = count_vectorizer.fit_transform(X_test.apply(lambda x : " ".join(x)))
predicted = MNB.predict(countvector_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)

print("accuracy:" ,accuracy_score*100)

#Confusion matrix
confusionmatrix=confusion_matrix(y_true=y_test,y_pred=predicted)
print("confusion matrix:")
print(confusionmatrix)

print("True Positive", confusionmatrix[0][0])
print("False Positive", confusionmatrix[0][1])
print("False Negative", confusionmatrix[1][0])
print("True Negative", confusionmatrix[1][1])

#implement Tfid vectorizer
tfidf = TfidfVectorizer(vocabulary=feature_list)
tfidf_train = tfidf.fit_transform(X_train.apply(lambda x : " ".join(x)))
tfidf_vector = pd.DataFrame(data = tfidf_train.toarray(),columns = tfidf.get_feature_names())
print(tfidf_vector)

#Multinomial naive bayes for TDIF vectorizer
MNB = MultinomialNB()
MNB.fit(tfidf_train,y_train)

tfidf_test = tfidf.fit_transform(X_test.apply(lambda x : " ".join(x)))
predicted = MNB.predict(tfidf_test)
accuracy_score = metrics.accuracy_score(predicted, y_test)

print("Tfidf Accuracy:" ,accuracy_score*100)

#TDIF confusion matrix
confusionmatrix=confusion_matrix(y_true=y_test,y_pred=predicted)
print("Tfidf confusion matrix:")
print(confusionmatrix)
