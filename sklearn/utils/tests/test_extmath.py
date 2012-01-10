import numpy as np
import scipy.sparse as sp

from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_array_almost_equal

from sklearn.utils.extmath import safe_sparse_dot

def test_sparse_dense_dot():
    a = np.random.random((5,4))
    a_csr = sp.csr_matrix(a)
    b = np.random.random((4,3))

    assert_array_almost_equal(np.dot(a, b),
                              safe_sparse_dot(a_csr,b))

def test_dense_sparse_dot():
    a = np.random.random((5,4))
    b = np.random.random((4,3))
    b_csr = sp.csr_matrix(b)

    assert_array_almost_equal(np.dot(a, b),
                              safe_sparse_dot(a, b_csr))


def test_sparse_dense_dot_out():
    a = np.random.random((5,4))
    a_csr = sp.csr_matrix(a)
    b = np.random.random((4,3))
    out = np.zeros((a.shape[0], b.shape[1]))

    out2 = safe_sparse_dot(a_csr, b, out=out)

    assert_array_almost_equal(np.dot(a, b), out)
    assert_true(id(out.data), id(out2.data))
