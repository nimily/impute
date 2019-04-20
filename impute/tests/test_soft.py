import numpy as np
import numpy.linalg as npl
import numpy.random as npr

from impute.soft import update


def run_tests():
    npr.seed(1)

    d, r = 200, 2

    u = npr.normal(size=(d, r))
    s = npr.uniform(0, 1, r)
    v = npr.normal(size=(r, d))

    x = u @ np.diag(s) @ v

    mask = npr.binomial(1, 0.2, (d, d))

    def proj(x):
        return np.multiply(x, mask)

    y = proj(x + npr.normal(0, 1, (d, d)))

    def loss(x, lambda_):
        return npl.norm(proj(x - y), 'f') + lambda_ * npl.norm(x, 'nuc')

    lambda_ = 0.1

    print(f'error: {loss(x, lambda_)}')

    z_old = np.zeros((d, d))
    print(loss(z_old, lambda_))

    for i in range(10):
        z_old = update(y, z_old, proj, 1, lambda_)
        print(loss(z_old, lambda_))

    print('test_soft.py: all tests passed.')


if __name__ == '__main__':
    run_tests()
