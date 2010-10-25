""" Naives Bayes classifiers.
"""

# Author: Vincent Michel <vincent.michel@inria.fr>
# License: BSD Style.
import numpy as np

from .base import BaseEstimator, ClassifierMixin

class GNB(BaseEstimator, ClassifierMixin):
    """
    Gaussian Naive Bayes (GNB)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.
    y : array, shape = [n_samples]
        Target vector relative to X

    Attributes
    ----------
    proba_y : array, shape = nb of classes
              probability of each class.
    theta : array of shape nb_class*nb_features
            mean of each feature for the different class
    sigma : array of shape nb_class*nb_features
            variance of each feature for the different class


    Methods
    -------
    fit(X, y) : self
        Fit the model

    predict(X) : array
        Predict using the model.

    predict_proba(X) : array
        Predict the probability of each class using the model.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from scikits.learn.naive_bayes import GNB
    >>> clf = GNB()
    >>> clf.fit(X, Y)
    GNB()
    >>> print clf.predict([[-0.8, -1]])
    [1]

    See also
    --------

    """
    def __init__(self):
        pass

    def fit(self, X, y):
        theta = []
        sigma = []
        proba_y = []
        unique_y = np.unique(y)
        for yi in unique_y:
            theta.append(np.mean(X[y==yi,:], 0))
            sigma.append(np.var(X[y==yi,:], 0))
            proba_y.append(np.float(np.sum(y==yi)) / np.size(y))
        self.theta = np.array(theta)
        self.sigma = np.array(sigma)
        self.proba_y = np.array(proba_y)
        self.unique_y = unique_y
        return self


    def predict(self, X):
        y_pred = self.unique_y[np.argmax(self.predict_proba(X),1)]
        return y_pred


    def predict_proba(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.unique_y)):
            jointi = np.log(self.proba_y[i])
            n_ij = - 0.5 * np.sum(np.log(np.pi*self.sigma[i,:]))
            n_ij -= 0.5 * np.sum( ((X - self.theta[i,:])**2) /\
                                    (self.sigma[i,:]),1)
            joint_log_likelihood.append(jointi+n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        proba = np.exp(joint_log_likelihood)
        proba = proba / np.sum(proba,1)[:,np.newaxis]
        return proba


# Author: Mathieu Blondel
#
# License: BSD Style.

import numpy as np

def to_1_of_K(y, y_min=None, y_max=None):
    """
    Convert a label vector y to a matrix Y such that Y[i,j]=1 means that sample
    i has class j.
    """
    if y_min is None: y_min = y.min()
    if y_max is None: y_max = y.max()

    n_samples = y.shape[0]
    Y = np.zeros((n_samples, y_max-y_min+1))
    for i, ind in enumerate(y):
        Y[i, ind-y_min] = 1

    return Y

def softmax(loga, k=-np.inf, out=None):
    """
    Compute the sotfmax function (normalized exponentials) without underflow.

    exp^a_i / \sum_j exp^a_j
    """
    if out is None: out = np.empty_like(loga).astype(np.float64)
    m = np.max(loga)
    logam = loga - m
    sup = logam > k
    inf = np.logical_not(sup)
    out[sup] = np.exp(logam[sup])
    out[inf] = 0.0
    out /= np.sum(out)
    return out

def logsum(loga, k=-np.inf):
    """
    Compute a sum of logs without underflow.

    \log \sum_c e^{b_c} = log [(\sum_c e^{b_c}) e^{-B}e^B]
                        = log [(\sum_c e^{b_c-B}) e^B]
                        = [log(\sum_c e^{b_c-B}) + B

    where B = max_c b_c
    """
    B = np.max(loga)
    logaB = aB = loga - B
    sup = logaB > k
    inf = np.logical_not(sup)
    aB[sup] = np.exp(logaB[sup])
    aB[inf] = 0.0
    return (np.log(np.sum(aB)) + B)


def loglikelihood(X, Y, p_w_c_log, p_c_log):
    n_samples, n_features = X.shape
    _, n_classes = Y.shape

    lik = 0.0

    ## labeled
    # log P(x|c)
    p_x_c_log = np.zeros((n_samples, n_classes), np.float64)
    for d,w in zip(*X.nonzero()):
        p_x_c_log[d,:] += p_w_c_log[w,:] * X[d,w]

    # add log P(c) + lop P(x|c) if x has label c
    for d,c in zip(*Y.nonzero()):
        lik += p_x_c_log[d,c]
        if p_c_log is not None:
            lik += p_c_log[c]

    return lik

def loglikelihood_u(X, Y, X_u, p_w_c_log, p_c_log):
    n_samples, n_features = X.shape
    n_samples_u, _ = X_u.shape
    _, n_classes = Y.shape

    lik = loglikelihood(X, Y, p_w_c_log, p_c_log)

    ## unlabeled
    # log P(x|c)
    p_x_c_log = np.zeros((n_samples_u, n_classes), np.float64)
    for d,w in zip(*X_u.nonzero()):
        p_x_c_log[d,:] += p_w_c_log[w,:] * X_u[d,w]

    # add log P(c)
    if p_c_log is not None:
        p_x_c_log += p_c_log[np.newaxis,:]

    for d in range(n_samples_u):
        lik += logsum(p_x_c_log[d,:], k=-10)

    return lik

def normalize_p_c(p_c, alpha):
    M = len(p_c)
    denom = M * alpha + np.sum(p_c)
    p_c += alpha
    p_c /= denom

def normalize_p_w_c(p_w_c, alpha):
    V, X = p_w_c.shape
    denoms = V * alpha + np.sum(p_w_c, axis=0)
    p_w_c += alpha
    p_w_c /= denoms[np.newaxis,:]

def generate_classifier(n_classes, n_features, length, n_labels=1):
    clf = MultinomialNaiveBayes()
    clf.length = length
    clf.n_labels = n_labels
    clf.p_c = np.random.random(n_classes)
    clf.p_c /= clf.p_c.sum()
    clf.p_w_c = np.random.random((n_features, n_classes))
    clf.p_w_c /= np.sum(clf.p_w_c, axis=0)
    return clf

class BaseNaiveBayes(object):

    def __init__(self, alpha=1):
        """
        Parameters
        ----------
        alpha : int
            Smoothing parameter (0 for no smoothing).
        """
        self.alpha = alpha
        self.use_prior = False
        self.p_w_c = None
        self.p_c = None
        self.length = None
        self.n_labels = None
        self.debug = False

    def fit_semi(self, X, Y, X_u, maxiter=50, eps=0.01):
        """
        Fit the model by EM.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training matrix, where n_samples is the number of samples and
            n_features is the number of features.
        Y : array, shape = [n_samples, n_classes]
            Label matrix, where n_samples is the number of samples and
            n_classes is the number of classes.
            Y[i,j] = 1 means that sample i belongs to class j.
            Y[i,j] = 0 means that sample i does not belong to class j.
        X_u : array-like, shape = [n_samples, n_features]
              Training matrix (unlabeled), where n_samples is the number of
              samples and n_features is the number of features.

        Returns
        -------
        self

        References
        ----------

        "Semi-Supervised Text Classification Using EM", by Nigam et al.
        """
        X = np.asarray(X)
        X_u = np.asarray(X_u)
        Y = np.asarray(Y)

        n_samples, n_classes = Y.shape
        n_samples_, n_features = X.shape
        n_samples_u, n_features_ = X_u.shape

        assert(n_samples == n_samples_)
        assert(n_features == n_features_)

        # Average document length
        if self.length is None:
            self.length = int(np.ceil((np.sum(X) + np.sum(X_u)) /
                              float(n_samples + n_samples_u)))

        # Average number of labels per example
        self.n_labels = int(np.ceil(np.mean(np.sum(Y, axis=1))))

        # compute counts for labeled data once for all
        self.fit(X, Y, normalize=False)
        if self.use_prior:
            p_c_l = np.array(self.p_c, copy=True)
            normalize_p_c(self.p_c, self.alpha)

        p_w_c_l = np.array(self.p_w_c, copy=True)
        # normalize to get initial classifier
        normalize_p_w_c(self.p_w_c, self.alpha)

        p_c_log = np.log(self.p_c) if self.p_c is not None else None
        lik = loglikelihood_u(X, Y, X_u, np.log(self.p_w_c), p_c_log)

        for iteration in range(1, maxiter+1):
            # E-step: find the probabilistic labels for unlabeled data
            if hasattr(self, "predict_proba"):
                Yu = self.predict_proba(X_u)
            else:
                Yu = to_1_of_K(self.predict(X_u))

            # M-step: train classifier with the union of
            #         labeled and unlabeled data
            self.fit(X_u, Yu, normalize=False, sparse=False)

            if self.use_prior:
                self.p_c += p_c_l
                normalize_p_c(self.p_c, self.alpha)

            self.p_w_c += p_w_c_l
            normalize_p_w_c(self.p_w_c, self.alpha)

            p_c_log = np.log(self.p_c) if self.p_c is not None else None
            lik_new = loglikelihood_u(X, Y, X_u, np.log(self.p_w_c), p_c_log)
            lik_diff = lik_new - lik
            lik = lik_new

            if lik_diff < eps:
                print "No more progress, stopping EM at iteration", iteration
                break

            if self.debug:
                print "Iteration", iteration
                print "L += %f" % lik_diff

        return self

    def _get_p_c(self):
        if self.p_c is None:
            n_classes = self.p_w_c.shape[1]
            return [1.0 / n_classes] * n_classes
        else:
            return self.p_c

    def sample(self, n, multi_label=False):
        """
        Generate samples from the model.

        Parameters
        ----------
        n : int
            Number of samples.
        k : int
            Length of a sample.
        """
        func = self.sample_multi_label_example if multi_label \
               else self.sample_example
        samples, labels = zip(*[func() for i in range(n)])
        return np.array(samples, dtype=int), np.array(labels, dtype=int)

    def sample_example(self):
        p_c = self._get_p_c()

        # pick a class with probability P(c)
        c = np.random.multinomial(1, p_c).argmax()

        # pick a document length
        k = np.random.poisson(self.length) or 1

        # generate a document of length k words
        sample = np.random.multinomial(k, self.p_w_c[:, c])

        return sample, c

    def sample_multi_label_example(self):
        n_features, n_classes = self.p_w_c.shape
        p_c = self._get_p_c()

        # pick a number of labels per document
        # FIXME: use a distribution
        n = self.n_labels

        # pick n classes
        classes = []
        while len(classes) != n:
            # pick a class with probability P(c)
            c = np.random.multinomial(1, p_c).argmax()

            if not c in classes:
                classes.append(c)

        classes = np.array(classes)
        Y = np.sum(to_1_of_K(classes, y_min=0, y_max=n_classes-1), axis=0)

        # pick a document length
        k = np.random.poisson(self.length) or 1

        # generate a document of length k words
        sample = np.zeros(n_features, dtype=int)
        for i in range(k):
            c = classes[np.random.randint(len(classes))]
            w = np.random.multinomial(1, self.p_w_c[:,c]).argmax()
            sample[w] += 1

        return sample, Y

    def score(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)

        if len(Y.shape) == 1:
            Y = to_1_of_K(Y)

        return loglikelihood(X, Y, np.log(self.p_w_c), np.log(self.p_c))


class MultinomialNaiveBayes(BaseNaiveBayes):
    """
    Multinomial Naive Bayes.

    Implements Maximum A Posteriori (MAP) estimation as well as
    semi-supervised learning by Expectation-Maximization (EM).

    Methods
    -------
    fit(X, Y) : self
        Fit the model by MAP.

    fit_semi(X, Y, X_u) : self
        Fit the model by EM.

    predict(X) : array
        Predict using the model.

    predict_proba(X): array
        Predict using the model.

    sample(n): array
        Generate samples from the model.

    """

    def __init__(self, alpha=1, use_prior=True):
        """
        Parameters
        ----------
        alpha : int
            Smoothing parameter (0 for no smoothing).
        use_prior: boolean
            Whether to use prior or not.
        """
        BaseNaiveBayes.__init__(self, alpha)
        self.use_prior = use_prior

    def fit(self, X, Y, normalize=True, sparse=True):
        """
        Fit the model by MAP.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training matrix, where n_samples is the number of samples and
            n_features is the number of features.
        Y : array, shape = [n_samples]
            Target vector.

            or

            array, shape = [n_samples, n_classes]
            Target matrix, where n_samples is the number of samples and
            n_classes is the number of classes.
            Y[i,j] = 1 means that sample i belong to class j.
            Y[i,j] = 0 means that sample i does not belong to class j.

            (fractional numbers are also possible)

        Returns
        -------
        self

        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        n_samples, n_features = X.shape

        if len(Y.shape) == 1:
            Y = to_1_of_K(Y)

        n_samples_, n_classes = Y.shape
        assert(n_samples == n_samples_)

        # Average document length
        if self.length is None:
            self.length = int(np.sum(X) / float(n_samples))

        # Average number of labels per example
        self.n_labels = int(np.ceil(np.mean(np.sum(Y, axis=1))))

        # P(c)
        if self.use_prior:
            self.p_c = np.sum(Y, axis=0).astype(np.float64)

        # P(w|c)
        self.p_w_c = np.zeros((n_features,n_classes), dtype=np.float64)

        if sparse:
            # faster when Y is sparse
            # select indices of documents that have class c
            for d,c in zip(*Y.nonzero()):
                # select indices of terms that are non-zero
                for w in np.flatnonzero(X[d,:]):
                    self.p_w_c[w,c] += X[d,w] * Y[d,c]
        else:
            # faster when Y is non-sparse
            for d,w in zip(*X.nonzero()):
                self.p_w_c[w,:] += X[d,w] * Y[d,:]

        if normalize:
            if self.use_prior:
                normalize_p_c(self.p_c, self.alpha)
            normalize_p_w_c(self.p_w_c, self.alpha)

        return self

    def p_x_c_log_all(self, X):
        n_classes = self.p_w_c.shape[1]
        n_samples, n_features = X.shape
        p_x_c_log = np.zeros((n_samples, n_classes), np.float64)
        p_w_c_log = np.log(self.p_w_c)

        # log P(x|c)
        for d,w in zip(*X.nonzero()):
            p_x_c_log[d,:] += p_w_c_log[w,:] * X[d,w]

        return p_x_c_log

    def predict_proba(self, X, k=-10):
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training matrix, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------

        Y : array, shape = [n_samples, n_classes]
            Probabilistic predictions.

        """
        n_samples, n_features = X.shape

        # log P(x|c)
        p_x_c_log = self.p_x_c_log_all(X)

        # add log P(c)
        if self.use_prior:
            p_x_c_log += np.log(self.p_c)[np.newaxis,:]

        # sotfmax(log P(x|c) + log P(c)) = P(c|x)
        for d in range(n_samples):
            softmax(p_x_c_log[d,:], k, out=p_x_c_log[d,:])

        return p_x_c_log

    def predict(self, X):
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training matrix, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------

        y : array, shape = [n_samples]
            Most likely labels.
        """
        # log P(x|c)
        p_x_c_log = self.p_x_c_log_all(X)

        # add log P(c)
        if self.use_prior:
            p_x_c_log += np.log(self.p_c)[np.newaxis,:]

        return p_x_c_log.argmax(axis=1)

class ComplementNaiveBayes(BaseNaiveBayes):
    """
    Complement Naive Bayes.

    Methods
    -------
    fit(X, Y) : self
        Fit the model by MAP.

    predict(X) : array
        Predict using the model.

    References
    ----------

    "Tackling the Poor Assumptions of Naive Bayes Text Classifiers",
    by Rennie et al.

    """

    def fit(self, X, Y, normalize=True, sparse=True):
        """
        Fit the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training matrix, where n_samples is the number of samples and
            n_features is the number of features.
        Y : array, shape = [n_samples]
            Target vector.

            or

            array, shape = [n_samples, n_classes]
            Target matrix, where n_samples is the number of samples and
            n_classes is the number of classes.
            Y[i,j] = 1 means that sample i belong to class j.
            Y[i,j] = 0 means that sample i does not belong to class j.

            (fractional numbers are accept)

        Returns
        -------
        self

        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples, n_features = X.shape

        if len(Y.shape) == 1:
            Y = to_1_of_K(Y)

        n_samples_, n_classes = Y.shape
        assert(n_samples == n_samples_)

        # Length
        if self.length is None:
            self.length = int(np.sum(X) / float(n_samples))

        # P(w|c)
        self.p_w_c = np.zeros((n_features,n_classes), dtype=np.float64)
        self.nc = np.zeros(n_classes, dtype=np.float64)

        ind = np.arange(n_classes)

        if sparse:
            # faster when Y is sparse
            # select indices of documents that have class c
            for d,c in zip(*Y.nonzero()):
                # select indices of terms that are non-zero
                for w in np.flatnonzero(X[d,:]):
                    self.p_w_c[w, ind != c] += X[d,w] * Y[d,c]
        else:
            # faster when Y is non-sparse
            for d,w in zip(*X.nonzero()):
                for c in range(n_classes):
                    self.p_w_c[w,ind != c] += X[d,w] * Y[d,c]

        if normalize:
            normalize_p_w_c(self.p_w_c, self.alpha)

        return self

    def p_x_c_log_all(self, X):
        n_classes = self.p_w_c.shape[1]
        n_samples, n_features = X.shape
        p_x_c_log = np.zeros((n_samples, n_classes), np.float64)
        p_w_c_log = np.log(self.p_w_c)
        p_w_c_log /= np.sum(np.abs(p_w_c_log), axis=0)

        # log P(x|c)
        for d,w in zip(*X.nonzero()):
            p_x_c_log[d,:] += p_w_c_log[w,:] * X[d,w]

        return p_x_c_log

    def predict(self, X):
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training matrix, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------

        y : array, shape = [n_samples]
            Most likely labels.
        """
        return self.p_x_c_log_all(X).argmin(axis=1)

