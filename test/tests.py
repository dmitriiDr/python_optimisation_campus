import unittest
from humble import *

class HumbleTests(unittest.TestCase):
	#export
	def test_add(self):
	    op = Matrix_Operations()
	
	    for n in [10, 100, 1000, 10000]:
	
	        print(f"{n}x{n} np.array")
	        matrix_A = op.init_matrix(n,n, fast=True)
	        matrix_B = op.init_matrix(n,n, fast=True)
	        op.add_matrix(matrix_A, matrix_B)
	        # print(f"{n}x{n} lists")
	        # matrix_A = op.init_matrix(n,n, fast=False)
	        # matrix_B = op.init_matrix(n,n, fast=False)
	        # op.add_matrix(matrix_A, matrix_B)

if __name__ == "__main__":
    unittest.main()