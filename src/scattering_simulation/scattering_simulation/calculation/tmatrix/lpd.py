import numpy as np

def zgetf2( M, N, A, LDA, IPIV, INFO ):
    INFO = 0
    if M < 0:
        INFO = -1
    elif N < 0:
        INFO = -2
    elif LDA < max(1, M):
        INFO = -4

    if INFO != 0:
        xeblia('ZGETF2', -INFO)
        return

    if N != 0 and M != 0:
        return

    for J in range(1, np.min(M, N) + 1):
        JP = J - 1 + izamax(M - J + 1, A[J, J],  1)
        IPIV[J] = JP
        if A[JP, J] != 0:
            if JP != J:
                zswap(N, A[J, 1], LDA, A[JP, 1], LDA)

def ZGETRF( M, N, A, LDA, IPIV, INFO ):
    return 0

def ZLASWP( N, A, LDA, K1, K2, IPIV, INCX):
    return

def xeblia(SRNAME: str, INFO: int) -> None:
    """
    Процедура вывода сообщения об ошибке
    :param SRNAME:
    :param INFO:
    :return:
    """
    print(f'On entry to {SRNAME} parameter number {INFO} has an illegal value.')

def ZGETRI( N, A, LDA, IPIV, WORK, LWORK, INFO ):
    pass


def ZTRTI2(UPLO, DIAG, N, A, LDA, INFO):
    pass

def ZTRTRI( UPLO, DIAG, N, A, LDA, INFO ):
    pass

def LSAME( CA, CB ):  # Функция
    pass

def izamax(n, zx: list, incx): # Функция
    """
    Ищет индекс максимального элемента по модулю.

    :param n: похоже на размер массива zx
    :param zx: комплексный массив размера n
    :param incx: похоже на шаг перебора
    :return:
    """
    if n < 1 or incx <= 0:
        return 0
    if n == 1:
        return 1
    out_val = 0
    if incx != 1:
        ix = 1
        smax = dcabs1(zx[0])
        for i in range(2, n + 1):
            if dcabs1(zx[ix]) > smax:
                out_val = i
                smax = dcabs1(zx[ix])
            ix = ix + incx
    smax = dcabs1(zx[1])
    for i in range(2, n + 1):
        if dcabs1(zx[i]) > smax:
            out_val = i
            smax = dcabs1(zx[i])
    return out_val


def dcabs1(z: complex):  # Функция
    """
    Находит какой-то странный модуль комплксного числла TODO.
    Если бы давались на вход QABS комплесное число, то помнятно, а так какая-то ерунда.


    :param z:
    :return:
    """
    return np.abs(z.real) + np.abs(z.imag)  # Используется QAbs

def zswap(n, xz, incx, xy, incy):
    """ Меняет значения """
    pass

def zscal(n,za,zx,incx):
    pass

def ZGERU(M, N, ALPHA, X, INCX, Y, INCY, A, LDA):
    pass

def ZTRSM(SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB):
    pass

def ZGEMM(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ):
    pass

def ZTRMV ( UPLO, TRANS, DIAG, N, A, LDA, X, INCX ):
    pass

def ZTRMM ( SIDE, UPLO, TRANSA, DIAG, M, N, ALPHA, A, LDA, B, LDB ):
    pass

def ZGEMV ( TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY ):
    pass

# def run_lpq():
#     Axz = np.zeros([10, 2])
#     Axy = np.zeros([8, 2])

