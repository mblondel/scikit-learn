import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from scikits.learn.naive_bayes import MultinomialNaiveBayes
from scikits.learn.naive_bayes import ComplementNaiveBayes
from scikits.learn.naive_bayes import generate_classifier, to_1_of_K

length = 100
n_samples = 100
n_classes = 5
n_features = 1000
n_train = 70
n_labeled = (10, 20, 30, 40, 50)

np.random.seed(0)

X, y = generate_classifier(n_classes, n_features, length).sample(n_samples)

X_train = X[:n_train]
X_test = X[n_train:]
y_train = y[:n_train]
y_test = y[n_train:]

Y_train = to_1_of_K(y_train)

def test_complement():
    y =  np.array([0, 0, 0, 1, 2])

    X = np.array([[3, 0, 1],
                 [2, 1, 0],
                 [2, 0, 0],
                 [0, 2, 1],
                 [1, 0, 3]])

    for sparse in (True, False):
        clf = ComplementNaiveBayes()
        clf.fit(X, y, normalize=False, sparse=sparse)
        # number of times word 0 appeared in classes other than 0
        assert_equal(clf.p_w_c[0,0], 1)
        # number of times word 1 appeared in classes other than 0
        assert_equal(clf.p_w_c[1,0], 2)
        # number of times word 2 appeared in classes other than 2
        assert_equal(clf.p_w_c[2,2], 2)

def _test_semi(NaiveBayes):
    for n in n_labeled:
        print "n_labeled", n

        clf = NaiveBayes()
        clf.fit(X_train[:n], Y_train[:n])
        pred = clf.predict(X_test)
        print "Supervised Learning"
        print "Accuracy: %d/%d" % (np.sum(pred == y_test), len(y_test))

        clf = NaiveBayes()
        clf.fit_semi(X_train[:n], Y_train[:n], X_train[n:])
        pred = clf.predict(X_test)
        print "Semi-Supervised Learning"
        print "Accuracy: %d/%d" % (np.sum(pred == y_test), len(y_test))

        print "-----"

def test_semi_multinomial():
    print "Multinomial Naive Bayes"
    _test_semi(MultinomialNaiveBayes)

def test_semi_complement():
    print "Complement Naive Bayes"
    _test_semi(ComplementNaiveBayes)
