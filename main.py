import os
import pickle
import argparse

if not os.path.exists('naive_bayes.pickle'):
    os.system('python build_model.py')

f = open('naive_bayes.pickle', 'rb')
naive_bayes = pickle.load(f)
f.close()

f = open('count_vector.pickle', 'rb')
count_vector = pickle.load(f)
f.close()

parser = argparse.ArgumentParser()
parser.add_argument('--email', required=True)

args = parser.parse_args()

f = open(args.email+'.txt')
iter_f = iter(f)
str = ""
for line in iter_f:
    str = str + line

x = count_vector.transform([str])
print('this email is predicted to be', naive_bayes.predict(x)[0])
print('the probability of spam is', naive_bayes.predict_proba(x)[0][1])