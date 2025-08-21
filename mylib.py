#LCG
def lcg(N=500, x0=0.1, coeff=4):

    x = x0
    L = []
    for _ in range(N):
        x = coeff * x * (1 - x)
        L.append(x)
    return L

    
a_cons = 1103515245
c_cons = 12345
m_cons = 32768
def ab(seed, n):
    x = seed
    A= []
    for i in range(n):
        x = (a_cons * x + c_cons) % m_cons
        A.append(x)  
    return A

seed = 1
N = 1000
L = ab(seed, N)

k = 5

#Gauss jordan
def gauss_jordan_solve(A, b, eps=1e-14):
    n = len(A)
    M = []
    for i in range(n):
        row = A[i][:]
        row.append(b[i])
        M.append(row)

    def swap_rows(M, i, j):
        M[i], M[j] = M[j], M[i]

    def scale_row(M, i, factor):
        M[i] = [x * factor for x in M[i]]

    def row_op(M, i, j, factor):
        M[i] = [x + factor * y for x, y in zip(M[i], M[j])]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot_row][col]) < eps:
            raise ValueError("Singular or nearly singular matrix.")
        if pivot_row != col:
            swap_rows(M, col, pivot_row)

        pivot = M[col][col]
        scale_row(M, col, 1.0 / pivot)

        for r in range(n):
            if r != col:
                factor = M[r][col]
                if abs(factor) > eps:
                    row_op(M, r, col, -factor)

    return [M[i][-1] for i in range(n)]

