import numpy as np
import numba as nb

@nb.jit(nopython=True)
def a_one(x):
    return(x+1)

@nb.jit(nopython=True, parallel=True)
def test():
    N = int(1e3)

    a = np.random.random((N,N)).astype(np.float64)
    b = np.ones(N, dtype=np.float64)
    c = np.zeros((N,N), dtype=np.float64)

    for i in nb.prange(N):
        a[i,i] = b[i]
    
    #assert((a.all == test.all))
    return(c)

if __name__ == '__main__':
    c = test()
    print(c.shape)

import numpy as np
from numba import njit

tre_n_arr = np.empty((3, 4))
tre_n_arr.fill(0)


@njit
def emb():
    tre_n_arr[0][0] = 12

emb()