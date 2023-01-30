import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ham_path = "ham"
files = os.listdir(ham_path)
ham_data = []
for file in files:
    f = open(ham_path+"/"+file)
    iter_f = iter(f)
    str = ""
    for line in iter_f:
        str = str + line
    ham_data.append(str)

spam_path = "spam"
files = os.listdir(spam_path)
spam_data = []
for file in files:
    f = open(spam_path+"/"+file, errors='ignore')
    iter_f = iter(f)
    str = ""
    for line in iter_f:
        str = str + line
    spam_data.append(str)

n_ham = len(ham_data)
n_spam = len(spam_data)

data_all = ham_data + spam_data
labels = ['ham']*n_ham + ['spam']*n_spam


count_vector = CountVectorizer(stop_words='english')
data_all = count_vector.fit_transform(data_all)

x_train, x_test, y_train, y_test = train_test_split(data_all, labels, random_state=1, test_size=0.1)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(x_train, y_train)
print(naive_bayes.score(x_train, y_train))
print(naive_bayes.score(x_test, y_test))
print(classification_report(y_test, naive_bayes.predict(x_test)))

import pickle
f = open('naive_bayes.pickle', 'wb')
pickle.dump(naive_bayes, f)
f.close()

f = open('count_vector.pickle', 'wb')
pickle.dump(count_vector, f)
f.close()
exit()