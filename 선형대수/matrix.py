def shape(a):
    num_rows = len(a)
    num_cols = len(a[0]) if a else 0
    return num_rows, num_cols


def get_row(a, i):
    return a[i]


def get_column(a, j):
    return [a_i[j]
            for a_i in a]


def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn](i, j)
            for j in range(num_cols)
            for i in range(num_rows)]


def is_diagonal(i, j):
    return 1 if i == j else 0

identity_matrix = make_matrix(5, 5, is_diagonal)



