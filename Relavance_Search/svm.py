
"""
Beating the Benchmark
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
import pandas as pd
import numpy as np
import scipy as sp
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
import sys

from sklearn.metrics.pairwise import linear_kernel


import xgboost as xgb

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

def two_features(fit_data, data):
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    result = []
    for i, row in data.iterrows():
        query = set(x.lower() for x in token_pattern.findall(row["query"]))
        title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        tmp = list(fit_data[i])
        print tmp
        if len(title) > 0:
            tmp.append(len(query.intersection(title))/len(title))
        else:
            tmp.append(0)

        if len(description) > 0:
            tmp.append(len(query.intersection(description))/len(description))
        else:
            tmp.append(0)
        result.append(tmp)
    return result

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

def feature_engineering(X, data):
    # add two features
    two_features = extract_features(data)
    features = []
    for i in range(len(two_features[0])):
        feature = [two_features[0][i], two_features[1][i]]
        features.append(feature)
    X =  np.append(X, features, 1)
    return X

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: python svm.py description"
        exit(1)
    print sys.argv[1]

    # Load the training file
    train = pd.read_csv('train.csv').fillna("")
    test = pd.read_csv('test.csv').fillna("")

    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

    #train = trian[ train['relevance_variance'] < 0.5 ]

    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

    # do some lambda magic on text columns
    #traindata = list(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
    #testdata = list(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

    # Fit TFIDF
    tfv.fit(traindata)
    X =  tfv.transform(traindata)
    X_test = tfv.transform(testdata)


    # Initialize SVD
    svd = TruncatedSVD(n_components=300)
    X = svd.fit_transform(X)
    X_test = svd.fit_transform(X_test)

    # Add two features
    X = feature_engineering(X, train)
    X_test = feature_engineering(X_test, test)


    # =====
    # xgboost try
    y = y - 1
    xg_train = xgb.DMatrix(X, label=y)
    xg_test = xgb.DMatrix(X_test, label=[1]*X_test.shape[0])
    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    param = {}
    param['objective'] = 'multi:softmax'
    param['num_class'] = 4
    num_round = 5
    bst = xgb.train(param, xg_train, num_round, watchlist)
    pred = bst.predict(xg_test)
    print pred
    exit(1)
    # =====

    # Initialize the standard scaler
    scl = StandardScaler()

    # We will use SVM here..
    svm_model = SVC()

    # Create the pipeline
    #clf = pipeline.Pipeline([('svd', svd),
    clf = pipeline.Pipeline([
                             ('scl', scl),
                    	     ('svm', svm_model)])

    # Create a parameter grid to search for best parameters for everything in the pipeline
    #param_grid = {'svd__n_components' : [200, 300, 400],
    param_grid = {
                  'svm__C': [10, 12]}

    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

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

    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("svm.csv", index=False)
