def read_matrix(filename):
    with open(filename, 'r') as f:
        print(f"file data {f.read()}")
        matrix = []
        for line in f:
            print(f"file data {line}")
            print(line)
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix

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

#cholesky iteration
def cholesky_decomposition(A):
    n = len(A)
    L = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k]*L[j][k] for k in range(j))
            if i == j:
                L[i][j] = (A[i][i] - s) ** 0.5
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L

def forward_substitution(L, b):
    n = len(L)
    y = [0.0]*n
    for i in range(n):
        s = sum(L[i][j]*y[j] for j in range(i))
        y[i] = (b[i] - s)/L[i][i]
    return y

def backward_substitution(L, y):
    n = len(L)
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        s = sum(L[j][i]*x[j] for j in range(i+1, n))
        x[i] = (y[i] - s)/L[i][i]
    return x

#jacobi iteration
def jacobi_iteration(A, b, tol=1e-6, max_iter=1000):
    n = len(b)
    x = [0.0] * n
    x_new = [0.0] * n

    for iteration in range(max_iter):
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]
            x_new[i] = (b[i] - s) / A[i][i]

        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        if diff < tol:
            return x_new, iteration + 1

        x = x_new[:]

    return x, max_iter
