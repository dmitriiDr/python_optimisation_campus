#export
def init_matrix(nrows, ncols):
    import numpy as np
    return [[np.random.randint(0, 100) for col in range(ncols)] for row in range(nrows)]#export
def add_matrix(matrix_a, matrix_b):

    if len(matrix_a) != len(matrix_b):
        raise ValueError('Matrices must have the same size')
        return None
    
    result = []
    for i in range(len(matrix_a)):
        if len(matrix_a[i]) != len(matrix_b[i]):
            raise ValueError('Rows in matrices must have the same length')
        result.append([a + b for a, b in zip(matrix_a[i], matrix_b[i])])
    return result# export
def mul_matrix(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError('Numer of columns in the first matrix is not the same as the number of rows in the second matrix')
        return None
    result =[]

    for i in range(len(matrix_a)):
        result.append([])
        for j in range(len(matrix_b[0])):
            result[i].append(0)
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result#export
from functools import wraps
import time
import numpy as np

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"It took {end_time - start_time} seconds")
        return result
    return wrapper#export
from numba import jit, njit, prange, vectorize#export
@njit(parallel=True)
# @vectorize(['int64(int64, int64)'], target='cpu')
def add_matrix_jit(matrix_a, matrix_b):
    result = np.empty_like(matrix_a)
    for i in prange(matrix_a.shape[0]):
        for j in prange(matrix_a.shape[1]):
            result[i, j] = matrix_a[i, j] + matrix_b[i, j]
    return result#export
class Matrix_Operations:


    def __init__(self):
        pass

    def init_matrix(self, nrows, ncols, fast):
        self.fast = fast
        self.nrows = nrows
        self.ncols = ncols
        import numpy as np

        if self.fast:
            return np.random.randint(0, 100, size=(nrows, ncols))
        else:
            return [[np.random.randint(0, 100) for col in range(ncols)] for row in range(nrows)]
        
        
    # @jit
    @timer
    def add_matrix(self, matrix_a, matrix_b):

        # if len(matrix_a) != len(matrix_b):
        #     raise ValueError('Matrices must have the same size')
        #     return None
        
        # result = []
        # for i in range(len(matrix_a)):
        #     if len(matrix_a[i]) != len(matrix_b[i]):
        #         raise ValueError('Rows in matrices must have the same length')
        #     result.append([a + b for a, b in zip(matrix_a[i], matrix_b[i])])
        return add_matrix_jit(matrix_a, matrix_b)
    
    # @jit
    # @timer
    def mul_matrix(self, matrix_a, matrix_b):
        if len(matrix_a[0]) != len(matrix_b):
            raise ValueError('Numer of columns in the first matrix is not the same as the number of rows in the second matrix')
            return None
        result =[]

        for i in range(len(matrix_a)):
            result.append([])
            for j in range(len(matrix_b[0])):
                result[i].append(0)
                for k in range(len(matrix_b)):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]
        return result
    
    def scalar_product(self, scalar, matrix):
        result = []
        for i in range(len(matrix)):
            result.append([])
            for j in range(len(matrix[i])):
                result[i].append(scalar * matrix[i][j])
        return result
    
    def transpose(self, matrix):
        return [list(row) for row in zip(*matrix)]
    
    def is_identity(self, matrix):
        if not matrix:
            raise IndexError('Matrix is epmty')
        # if len(matrix) != len(matrix[0]):
            return False
        rows = len(matrix)
        for i in range(rows):
            for j in range(rows):
                if i == j and matrix[i][j] != 1:
                    return False
                if i != j and matrix[i][j] != 0:
                    return False
        return True
