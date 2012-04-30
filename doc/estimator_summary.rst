.. include:: includes/big_toc_css.rst

.. _estimator_summary:

=========================
Estimators at a glance
=========================

Classification
--------------

============================  =====  ======  ===========  ==========
Estimator                     Dense  Sparse  Multi-class  Multi-task
============================  =====  ======  ===========  ==========
:class:`.GaussianNB`          Yes    Yes     Built-in     No
:class:`.LogisticRegression`  Yes    Yes     One-Vs-Rest  No
:class:`.MultinomialNB`       Yes    Yes     Built-in     No
:class:`.NuSVC`               Yes    Yes     One-Vs-One   No
:class:`.Perceptron`          Yes    Yes     One-Vs-Rest  No
:class:`.RidgeClassifier`     Yes    Yes     One-Vs-Rest  No
:class:`.RidgeClassifierCV`   Yes    Yes     One-Vs-Rest  No
:class:`.SGDClassifier`       Yes    Yes     One-Vs-Rest  No
:class:`.SVC`                 Yes    Yes     One-Vs-One   No
============================  =====  ======  ===========  ==========

Regression
----------

=====================================  =====  ======  ==========
Estimator                              Dense  Sparse  Multi-task
=====================================  =====  ======  ==========
:class:`.ARDRegression`                Yes    No      No
:class:`.BayesianRidge`                Yes    No      No
:class:`.ElasticNet`                   Yes    No      No
:class:`.ElasticNetCV`                 Yes    No      No
:class:`.Lars`                         Yes    No      No
:class:`.LarsCV`                       Yes    No      No
:class:`.Lasso`                        Yes    No      No
:class:`.LassoCV`                      Yes    No      No
:class:`.LassoLars`                    Yes    No      No
:class:`.LassoLarsCV`                  Yes    No      No
:class:`.LassoLarsIC`                  Yes    No      No
:class:`.LinearRegression`             Yes    Yes     Yes
:class:`.NuSVR`                        Yes    Yes     No
:class:`.OrthogonalMatchingPursuit`    Yes    No      Yes
:class:`.Ridge`                        Yes    Yes     Yes
:class:`.RidgeCV`                      Yes    Yes     Yes
:class:`.SGDRegressor`                 Yes    Yes     Yes
:class:`.SVR`                          Yes    Yes     No
=====================================  =====  ======  ==========
