from bs4 import BeautifulSoup
from nltk.stem.porter import *
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import math
import numpy as np
import pandas as pd
import re

# declarations
stemmer = PorterStemmer()
sw=[]
s_data = []
t_data = []
t_queries = []
t_labels = []
t_labelsf = []
#stopwords tweak
ML_STOP_WORDS = ['http','www','img','border','color','style','padding','table','font','inch','width','height']
ML_STOP_WORDS += list(text.ENGLISH_STOP_WORDS)
for stw in ML_STOP_WORDS:
    sw.append("z"+str(stw))
ML_STOP_WORDS += sw
for i in range(len(ML_STOP_WORDS)):
    ML_STOP_WORDS[i]=stemmer.stem(ML_STOP_WORDS[i])

def ML_TEXT_CLEAN(f2,f3):
    if len(f2)<3:
        f2="feature2null"
    if len(f3)<3:
        f3="feature3null"
    tx = BeautifulSoup(f3)
    tx1 = [x.extract() for x in tx.findAll('script')]
    tx = tx.get_text(" ").strip()

    tx2 = BeautifulSoup(f2)
    s = (" ").join(["z"+ z for z in tx2.get_text(" ").split(" ")]) + " " + tx
    s = re.sub("[^a-zA-Z0-9]"," ", s)
    s = re.sub("[0-9]{1,3}px"," ", s)
    s = re.sub(" [0-9]{1,6} |000"," ", s)
    s = (" ").join([stemmer.stem(z) for z in s.split(" ") if len(z)>2])
    s = s.lower()
    return s

def vote_ensemble(preds):
    n = len(preds)
    m = len(preds[0])
    pred = []
    from collections import Counter
    for i in range(m):
        labels = []
        for j in range(n):
            labels.append(int(preds[j][i]))
        pred.append( Counter(labels).most_common()[0][0] )
    return pred

def average_ensemble(preds):
    n = len(preds)
    m = len(preds[0])
    pred = []
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += int(preds[j][i])
        pred.append( round(sum*1.0/n, 0) )
    return pred

#load data
train = pd.read_csv("train.csv").fillna(" ")
test  = pd.read_csv("test.csv").fillna(" ")

for i in range(len(train.id)):
    s = ML_TEXT_CLEAN(train.product_title[i], train.product_description[i])
    s_data.append((train["query"][i], s, str(train["median_relevance"][i])))
for i in range(len(test.id)):
    s = ML_TEXT_CLEAN(test.product_title[i], test.product_description[i])
    t_data.append((test["query"][i], s, test.id[i]))
    if test["query"][i] not in t_queries:
        t_queries.append(test["query"][i])

df1 = pd.DataFrame(s_data)
df2 = pd.DataFrame(t_data)
for tq in t_queries:
    df1_s = df1[df1[0]==tq]
    df2_s = df2[df2[0]==tq]
    #Naive Bayes
    print tq
    print "Naive Bayes"
    clf = MultinomialNB(alpha=0.01)
    v = TfidfVectorizer(use_idf=True,min_df=0,ngram_range=(1,6),lowercase=True,sublinear_tf=True,stop_words=ML_STOP_WORDS)
    clf.fit(v.fit_transform(df1_s[1]), df1_s[2])
    t_labels_nb = clf.predict(v.transform(df2_s[1]))
    #SDG
    print "SGD"
    clf = Pipeline([('v',TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 6), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = ML_STOP_WORDS)), ('sdg', SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1, eta0=0.0, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True, verbose=0, warm_start=False))])
    clf.fit(df1_s[1], df1_s[2])
    t_labels_sdg = clf.predict(df2_s[1])
    #SVD/Standard Scaler/SVM
    print "SVD SCL SVM"
    clf = Pipeline([('v',TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 6), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = ML_STOP_WORDS)), ('svd', TruncatedSVD(n_components=100)),  ('scl', StandardScaler()), ('svm', SVC(C=10))])
    clf.fit(df1_s[1], df1_s[2])
    t_labels_sv_ = clf.predict(df2_s[1])
    #Decision Tree
    print "DT..."
    clf = Pipeline([('v',TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 6), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = ML_STOP_WORDS)), ('dtc', DecisionTreeClassifier(random_state=0))])
    clf.fit(df1_s[1], df1_s[2])
    t_labels_dtc = clf.predict(df2_s[1])
    #KNeighbors
    print "KNN..."
    clf = Pipeline([('v',TfidfVectorizer(max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 6), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = ML_STOP_WORDS)), ('kn', KNeighborsClassifier(n_neighbors=3))])
    clf.fit(df1_s[1], df1_s[2])
    t_labels_kn = clf.predict(df2_s[1])

    print "labels..."
    for i in range(len(t_labels_nb)):
        t_labels1 = list(df2_s[2])
        t_labelsf.append((t_labels1[i],t_labels_nb[i],t_labels_sdg[i],t_labels_sv_[i],t_labels_dtc[i],t_labels_kn[i]))

df3 = pd.DataFrame(t_labelsf)
df3 = df3.sort([0])

preds = []
for i in range(5):
    preds.append( list(df3[i+1]))

pred_vote = vote_ensemble(preds)
submission = pd.DataFrame({"id": test.id, "prediction": pred_vote})
submission.to_csv("ensemble5models_vote.csv", index=False)

pred_average = average_ensemble(preds)
submission = pd.DataFrame({"id": test.id, "prediction": pred_average})
submission.to_csv("ensemble5models_average.csv", index=False)
