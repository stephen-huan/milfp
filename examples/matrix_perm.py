import milfp as mip
import numpy as np
"""
matrix permutation problem:
does there exist permutation matrices P, Q
such that PAQ = M for given matrices A, M?
"""

def solve(A: np.array, M: np.array) -> bool:
    """ Computes if there exists permutation matrices such that PAQ = M. """
    ### model and variables
    m = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
    P, Q = (m.add_var_tensor(A.shape, name=f"matrix_{i}", var_type=mip.BINARY)
            for i in range(2))

    ### constraints
    # ensure matrices are permutation matrices
    for matrix in [P, Q]:
        for row in matrix:
            m += mip.xsum(row) == 1
        for col in matrix.T:
            m += mip.xsum(col) == 1

    ### objective
    # find the permutation matrices which minimize distance to M
    m.objective = mip.xsum(x*x for x in (P@A@Q - M).ravel())

    status = m.optimize()
    return abs(m.objective_value) < 10**-3 # P, Q exists iff 0 distance

if __name__ == "__main__":
    A = np.array([[0, 2, 2],
                  [2, 0, 1],
                  [0, 1, 1]])
    # is a row/column permutation of A
    M1 = np.array([[0, 1, 2],
                   [2, 2, 0],
                   [1, 1, 0]])
    # is not a permutation of A
    M2 = np.array([[0, 1, 2],
                   [2, 2, 0],
                   [0, 1, 1]])

    print(solve(A, M1))
    print(solve(A, M2))

