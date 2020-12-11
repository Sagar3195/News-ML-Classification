import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import nltk
##Loading dataset
data = pd.read_csv("news.csv")

print(data.head())

print(data.shape)

##Let's checking missing values in dataset
print(data.isnull().sum())

### We can see that there is no missing values in dataset.

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

corpus = []
for i in range(len(data)):
  text = re.sub('[^a-zA-Z]', ' ', data['text'][i])
  text = text.lower()
  text = text.split()
  text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
  text = ' '.join(text)
  corpus.append(text)

#print(corpus)

print(len(corpus))

##Now creating the model for Tf-Idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 3000, ngram_range = (1,3))

### Now we create independent variable
X = vectorizer.fit_transform(corpus)

X = X.toarray()

print(X.shape)

print(X.size)

print(data['label'].value_counts())
##Now we do label enconding
data['label'] = data['label'].replace({"REAL": 0, "FAKE":1})

print(data['label'].value_counts(normalize = True))

###Dependent variable
y = data['label']

print(y.shape)

### Now splitting dataset into training data & testing data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)

##Here we can see that combinations of 2 & 3 words bcoz of ngram_range parameter

print(vectorizer.get_feature_names()[:30])

## Now we create classification model 
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

##Now we train the model using fit method and predict the model

classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

##Let's check the performan of the model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
accuracy = accuracy_score(y_test, prediction)
print("Accuracy of the model: ", accuracy)

print("Confusion matrix of the model: \n", confusion_matrix(y_test, prediction))

print("Classification report of the model:\n", classification_report(y_test, prediction))

import joblib

#joblib.dump(classifier, 'news_model.pkl')

#joblib.dump(vectorizer, 'transform.pkl')





