
"""
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text

from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_absolute_error

from gensim.models import ldamodel
from gensim import corpora

# array declarations
sw=[]
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
    sw.append("q"+stw)
    sw.append("z"+stw)
stop_words = text.ENGLISH_STOP_WORDS.union(sw)


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

def svm_one(train, test):
    print "svm_one"

    # we dont need ID columns
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)


    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

    # Fit TFIDF
    tfv.fit(traindata)
    X =  tfv.transform(traindata)
    X_test = tfv.transform(testdata)




    ## add two features
    #two_features = extract_features(train)
    #features = []
    #for i in range(len(two_features[0])):
    #    feature = [two_features[0][i], two_features[1][i]]
    #    features.append(feature)
    #X = sp.sparse.csr_matrix( np.append(X.toarray(), features, 1) )
    #two_features = extract_features(test)
    #features = []
    #for i in range(len(two_features[0])):
    #    feature = [two_features[0][i], two_features[1][i]]
    #    features.append(feature)
    #X_test = sp.sparse.csr_matrix( np.append(X_test.toarray(), features, 1) )


    # Initialize SVD
    svd = TruncatedSVD()

    # Initialize the standard scaler
    scl = StandardScaler()

    # We will use SVM here..
    svm_model = SVC()

    # GradientBoostingClassifier
    #gbm = GradientBoostingClassifier()

    # Create the pipeline
    clf = pipeline.Pipeline([('svd', svd),
    						 ('scl', scl),
                    	     ('svm', svm_model)])

    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components' : [200, 400],
                  'svm__C': [10, 12]}

    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    #kappa_scorer = metrics.make_scorer(accuracy_score, greater_is_better = True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    # Fit Grid Search Model
    model.fit(X, y)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Get best model
    best_model = model.best_estimator_

    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(X,y)
    preds = best_model.predict(X_test)

    #scores = cross_validation.cross_val_score(best_model, X, y, cv=10)
    #print scores

    # Add two features for ensemble
    #preds = two_features_ensemble(train, test, preds)
    return preds

def svm_two(train, test, stack=0, true_test=None):
    print "svm_two"
    #remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
    stemmer = PorterStemmer()
    ## Stemming functionality
    class stemmerUtility(object):
        """Stemming functionality"""
        @staticmethod
        def stemPorter(review_text):
            porter = PorterStemmer()
            preprocessed_docs = []
            for doc in review_text:
                final_doc = []
                for word in doc:
                    final_doc.append(porter.stem(word))
                    #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
                preprocessed_docs.append(final_doc)
            return preprocessed_docs

    s_data = []
    s_labels = []
    t_data = []
    t_labels = []

    #for i in range(len(train.id)):
    for i in list(train.index):
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train["product_title"][i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train["product_description"][i]).get_text(" ")
        #s=(" ").join([z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join([z for z in BeautifulSoup(train["product_title"][i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train["product_description"][i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        #s_data.append(s.split(' '))
        s_labels.append(str(train["median_relevance"][i]))

    if stack == 1:
        t_labels = []

    #for i in range(len(test.id)):
    for i in list(test.index):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test["product_title"][i]).get_text().split(" ")]) + " " + BeautifulSoup(test["product_description"][i]).get_text()
        #s=(" ").join([z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join([z for z in BeautifulSoup(test["product_title"][i]).get_text().split(" ")]) + " " + BeautifulSoup(test["product_description"][i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
        #t_data.append(s.split(' '))

        if stack == 1:
            t_labels.append( str(test['median_relevance'][i]) )

    if stack == 1:
        true_test_data = []
        for i in list(true_test.index):
            s=(" ").join(["q"+ z for z in BeautifulSoup(true_test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(true_test["product_title"][i]).get_text().split(" ")]) + " " + BeautifulSoup(true_test["product_description"][i]).get_text()
            s=re.sub("[^a-zA-Z0-9]"," ", s)
            s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
            true_test_data.append(s)


    tfv = TfidfVectorizer(min_df=3, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')

    # Fit TFIDF
    tfv.fit(s_data)
    X =  tfv.transform(s_data)
    X_test = tfv.transform(t_data)

    # LDA, topic model
    #dictionary = corpora.Dictionary(s_data)
    #corpus = [dictionary.doc2bow(text) for text in s_data]
    #lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=100)
    #corpus_test = [dictionary.doc2bow(text) for text in t_data]
    #X = lda[corpus]
    #X_test = lda[corpus_test]

    ## add two features
    #two_features = extract_features(train)
    #features = []
    #for i in range(len(two_features[0])):
    #    feature = [two_features[0][i], two_features[1][i]]
    #    features.append(feature)
    #X = sp.sparse.csr_matrix( np.append(X.toarray(), features, 1) )
    #two_features = extract_features(test)
    #features = []
    #for i in range(len(two_features[0])):
    #    feature = [two_features[0][i], two_features[1][i]]
    #    features.append(feature)
    #X_test = sp.sparse.csr_matrix( np.append(X_test.toarray(), features, 1) )

    if stack == 1:
        entire_data = []
        entire_data.extend(s_data)
        entire_data.extend(t_data)
        entire_X = tfv.transform(entire_data)
        entire_labels = []
        entire_labels.extend(s_labels)
        entire_labels.extend(t_labels)
        true_test_X = tfv.transform(true_test_data)


    #create sklearn pipeline, fit all, and predit test data
    clf = Pipeline([
    ('svd', TruncatedSVD(algorithm='randomized', n_iter=5, random_state=None, tol=0.0)),
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
    ('svm', SVC(kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])


    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components' : [200, 400],
                  'svm__C': [10, 12]}
                  #'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer, verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    # Fit Grid Search Model
    model.fit(X, s_labels)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # Get best model
    best_model = model.best_estimator_

    best_model.fit(X, s_labels)
    t_pred = best_model.predict(X_test)

    if stack == 1:
        best_model.fit(X_test, t_labels)
        s_pred = best_model.predict(X)

        best_model.fit(entire_X, entire_labels)
        pred = best_model.predict(true_test_X)
        return [s_pred, t_pred, pred]

    return t_pred

# model two features
def last_stage(feature, label, test):
    clf = Pipeline([
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
    ('svm', SVC(kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    clf.fit(feature, label)
    pred = clf.predict(test)
    return pred

def extract_features(data):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    query_tokens_in_title = []
    query_tokens_in_description = []
    for i, row in data.iterrows():
        query = set(x.lower() for x in token_pattern.findall(row["query"]))
        title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        if len(title) > 0:
            query_tokens_in_title.append( len(query.intersection(title))/len(title))
        else:
            query_tokens_in_title.append(0)

        if len(description) > 0:
            query_tokens_in_description.append( len(query.intersection(description))/len(description))
        else:
            query_tokens_in_description.append(0)
    return [query_tokens_in_title, query_tokens_in_description]


import random
# random generate label by variance
def random_label(train, K):
    train_set = []
    for i in range(K):
        train_set.append( train.copy() )

    label_set = list(np.unique(train['median_relevance']))
    max_variance = train['relevance_variance'].max()

    for i in list(train.index):
        print "Preprocessing " + str(i) + "th train data"
        num_change = int(train['relevance_variance'][i] * K/max_variance)
        tmp_label = train['median_relevance'][i]
        if tmp_label == 1:
            tmp_label_set = [2]
        elif tmp_label == 2:
            tmp_label_set = [1,3]
        elif tmp_label == 3:
            tmp_label_set = [2,4]
        elif tmp_label == 4:
            tmp_label_set = [3]
        else:
            pass

        tmp_len = len(tmp_label_set)
        for j in range(num_change):
            train_set[j]['median_relevance'][i] = tmp_label_set[random.randint(0,tmp_len-1)]

    for i in range(K):
        print train_set[i]['median_relevance'].value_counts()
    return train_set

def svm_average(train,test):
    t_labels = svm_two(train, test)
    preds = svm_one(train, test)

    import math
    p3 = []
    for i in range(len(preds)):
        x = (int(t_labels[i]) + int(preds[i]))/2
        x = math.floor(x)
        p3.append(int(x))
    return p3


def average_submission():
    #load data
    train = pd.read_csv("train.csv").fillna("")
    test  = pd.read_csv("test.csv").fillna("")
    idx = test.id.values.astype(int)

    # we select low variance samples
    #train = train[ train['relevance_variance'] < 0.5 ]

    t_labels = svm_two(train, test)
    preds = svm_one(train, test)

    import math
    p3 = []
    for i in range(len(preds)):
        #x = (int(t_labels[i]) + int(preds[i]))/2
        #x = math.floor(x)
        #p3.append(int(x))
        x =math.sqrt((int(t_labels[i]))*int(preds[i]))
        x = round(x)
        p3.append(int(x))

    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": p3})
    submission.to_csv("average_ensemble.csv", index=False)

def vote_ensemble(preds):
    n = len(preds)
    m = len(preds[0])
    pred = []
    from collections import Counter
    for i in range(m):
        labels = []
        for j in range(n):
            labels.append(preds[j][i])
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
        pred.append( int(sum/n) )
    return pred

def variance_submission():
    #load data
    train = pd.read_csv("train.csv").fillna("")
    test  = pd.read_csv("test.csv").fillna("")
    idx = test.id.values.astype(int)

    K = 10
    train_set = random_label(train, K)
    preds = []
    for i in range(K):
        print "Training " + str(i) + "th model......"
        pred = svm_average(train_set[i], test)
        submission = pd.DataFrame({"id": idx, "prediction": pred})
        submission.to_csv("variance_ensemble" + str(i) + ".csv", index=False)
        preds.append(pred)
        print str(i) + "th prediction done!!!"

    # ensemble by vote and average
    pred_vote = vote_ensemble(preds)
    submission = pd.DataFrame({"id": idx, "prediction": pred_vote})
    submission.to_csv("variance_ensemble_vote.csv", index=False)

    pred_average = average_ensemble(preds)
    submission = pd.DataFrame({"id": idx, "prediction": pred_average})
    submission.to_csv("variance_ensemble_average.csv", index=False)


#####
# Split the train set in 2 parts: train_a and train_b
# Fit a first-stage model on train_a and create predictions for train_b
# Fit the same model on train_b and create predictions for train_a
# Finally fit the model on the entire train set and create predictions for the test set.
# Now train a second-stage stacker model on the probabilities from the first-stage model(s)
#####
def stack_submission():
    #load data
    train = pd.read_csv("train.csv").fillna("")
    test  = pd.read_csv("test.csv").fillna("")
    idx = test.id.values.astype(int)
    y = list(train['median_relevance'].values)

    A_train, B_train, A_test, B_test = cross_validation.train_test_split(train, y, test_size=0.5, random_state=0)

    pred = svm_two(A_train, B_train, 1, test)


    feature_train = []
    feature_test = []

    two_features = extract_features(A_train)
    for i in range( len(A_train.index) ):
        feature = [pred[0][i], two_features[0][i], two_features[1][i]]
        feature_train.append(feature)

    two_features = extract_features(B_train)
    for i in range( len(B_train.index) ):
        feature = [pred[1][i], two_features[0][i], two_features[1][i]]
        feature_train.append(feature)

    two_features = extract_features(test)
    for i in range( len(test.index) ):
        feature = [pred[2][i], two_features[0][i], two_features[1][i]]
        feature_test.append(feature)

    A_test.extend(B_test)
    pred = last_stage(feature_train, A_test, feature_test)

    submission = pd.DataFrame({"id": idx, "prediction": pred})
    submission.to_csv("stack_ensemble.csv", index=False)

def single_submission():
    #load data
    train = pd.read_csv("train.csv").fillna("")
    test  = pd.read_csv("test.csv").fillna("")
    idx = test.id.values.astype(int)

    # we select low variance samples
    #train = train[ train['relevance_variance'] < 0.5 ]

    pred = svm_one(train, test)

    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": pred})
    submission.to_csv("single.csv", index=False)


if __name__ == '__main__':
    print "@submission"
    #variance_submission()
    #stack_submission()
    average_submission()
    #single_submission()
