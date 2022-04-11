import numpy as np
from .ampld_par import *


def GAUSS(N, IND1, IND2, Z, W):
    A = 1.
    B = 2.
    C = 3.
    IND = np.mod(N, 2)
    K = N / 2 + IND
    F = float(N)

    for I in range(K):
        M = N - 1 - I
        if I == 0:
            X = A - B / ((F + A) * F)
        if I == 1:
            X = (Z[N - 1] - A) * 4 + Z[N - 1]
        if I == 2:
            X = (Z[N - 2] - Z[N - 1]) * 1.6 + Z[N - 2]
        if I > 2:
            X = (Z[M + 1] - Z[M + 2]) * C + Z[M + 3]
        if I == K and IND == 0:
            X = 0
        NITER = 0
        CHECK = 1e-16

        while True:
            PB = 1
            NITER = NITER + 1
            if NITER > 100:
                CHECK = CHECK * 10
            PC = X
            DJ = A

            for J in range(1, N):
                DJ = DJ + 1
                PA = PB
                PB = PC
                PC = X * PB + (X * PB - PA) * (DJ - A) / DJ

            PA = A / ((PB - X * PC) * F)
            PB = PA * PC * (A - X * X)
            X = X - PB
            if abs(PB) <= CHECK * abs(X):
                break
        Z[M] = X
        W[M] = PA * PA * (A - X * X)
        if IND1 == 0:
            W[M] = B * W[M]

        if not (I == K and IND == 1):
            Z[I] = - Z[M]
            W[I] = W[M]
    if IND2 == 1:
        print(f'***  POINTS AND WEIGHTS OF GAUSSIAN QUADRATURE FORMULA OF ', N, '-TH ORDER')

        for I in range(K):
            ZZ = -Z[I]
            print(f'  X[{I}] = {ZZ}, W[{I}] = {W[I]}')
        print(f' GAUSSIAN QUADRATURE FORMULA OF {N}-TH ORDER IS USED')

    if IND1 != 0:
        for I in range(N):
            Z[I] = (A + Z[I]) / B

    return Z, W

def SAREA(D, RAT):
    UNUSED = RAT
    if D < 1:
        E = np.sqrt(1 - D * D)
        R = 0.5 * (D**(2./3.) + D**(-1/3) * np.asin(E)/E)
        R = np.sqrt(R)
        RAT = 1 / R
        return RAT
    E = np.sqrt(1. - 1. / (D * D))
    R = 0.25 * (2. * D**(2./3.) + D**(-4/3) * np.log((1 + E) / (1 - E)) / E)
    R = np.sqrt(R)
    RAT = 1 / R
    return RAT

def SURFCH (N, E, RAT):
    UNUSED = RAT
    X = np.zeros(60)
    W = np.zeros(60)
    DN = float(N)
    E2 = E * E
    EN = E * DN
    NG = 60
    X, W = GAUSS(NG, 0, 0, X, W)
    S = 0
    V = 0

    for I in range(NG):
        XI = X[I]
        DX = np.acos(XI)
        DXN = DN * DX
        DS = np.sin(DX)
        DSN = np.sin(DXN)
        DCN = np.cos(DXN)
        A = 1 + E*DCN
        A2 = A*A
        ENS = EN*DSN
        S = S+W[I]*A*np.sqrt(A2+ENS*ENS)
        V = V+W[I]*(DS*A+XI*ENS)*DS*A2

    RS=np.sqrt(S*0.5)
    RV=(V*3./4.)**(1./3.)
    RAT=RV/RS
    return RAT

def SAREAC(EPS, RAT):
    UNUSED = RAT
    RAT = (1.5 / EPS)**(1/3)
    RAT = RAT / np.sqrt((EPS + 2.) / (2. * EPS))
    return RAT

def DROP(RAT):
    C = [-0.0481, 0.0359, -0.1263, 0.0244, 0.0091, -0.0099, 0.0015, 0.0025, -0.0016, 0.0002, 0.001]
    NG = 60
    NC = 10
    X = np.zeros(NG)
    W = np.zeros(NG)
    X, W = GAUSS(NG, 0, 0, X, W)
    S = 0
    V = 0

    for I in range(NG):
        XI = np.cos(X[I])
        WI = W[I]
        RI = 1 + C[0]
        DRI = 0
        for N in range(1, NC + 1):
            XIN = XI * N
            RI = RI + C[N] * np.cos(XIN)
            DRI = DRI - C[N] * N * np.sin(XIN)
        SI = np.sin(XI)
        CI = X[I]
        RISI = RI * SI
        S = S + WI * RI * np.sqrt(RI**2 + DRI**2)
        V = V + WI * RI * RISI * (RISI - DRI * CI)

    RS = np.sqrt(S * 0.5)
    RV = (V * 3. / 4.)**(1./3.)

    if np.abs(RAT - 1.) > 1.e-8:
        RAT = RV / RS
    ROV = 1 / RV
    print(f'r_0/r_ev = {ROV}')
    for step in range(NC + 1):
        print(f'c_{step} = {C[N]}')

    return RAT

def CONST(NGAUSS, NMAX, MMAX, P, X, W, AN, ANN, S, SS, NP, EPS):
    X1 = np.zeros(NPNG1)
    W1 = np.zeros(NPNG1)
    X2 = np.zeros(NPNG1)
    W2 = np.zeros(NPNG1)
    DD = np.zeros(NPN1)

    for N in range(NMAX):
        NN = N * (N - 1)
        AN[N] = NN
        D = np.sqrt(2 * N + 1) / NN
        DD[N] = D
        for N1 in range(N):
            DDD = D * DD[N1] * 0.5
            ANN[N, N1] = DDD
            ANN[N1, N] = DDD
    NG = 2 * NGAUSS
    if NP != -2:
        X, W = GAUSS(NG, 0, 0, X, W)
    else:
        NG1 = NGAUSS / 2.
        NG2 = NGAUSS - NG1
        XX = -np.cos(np.atan(EPS))
        X1, W1 = GAUSS(NG1, 0, 0, X1, W1)
        X2, W2 = GAUSS(NG2, 0, 0, X2, W2)
        for I in range(NG1):
            W[I] = 0.5 * (XX + 1) * W1[I]
            X[I] = 0.5 * (XX + 1) * X1[I] + 0.5 * (XX - 1)
        for I in range(NG2):
            W[I + NG1] = -0.5 * XX * W2[I]
            X[I + NG1] = -0.5 * XX * X2[I] + 0.5 * XX
        for I in range(NGAUSS):
            W[NG - I + 1] = W[I]
            X[NG - I + 1] = -X[I]
    for I in range(NGAUSS):
        Y = X[I]
        Y = 1. / (1. - Y**2)
        SS[I] = Y
        SS[NG - I + 1] = Y
        Y = np.sqrt(Y)
        S[I] = Y
        S[NG - I + 1] = Y
    return X, W, AN, ANN, S, SS


def VARY (LAM, MRR, MRI, A, EPS, NP, NGAUSS, X, P, PPI, PIR, PII, R, DR, DDR, DRR, DRI, NMAX):
    NG = NGAUSS * 2
    if NP > 0:
        R, DR = RSP2(X, NG, A, EPS, NP, R, DR)

def RSP2(X, NG, REV, EPS, N, R, DR):
    DNP = N
    DN = DNP**2.
    DN4 = DN * 4.
    EP = EPS**2
    A = 1. + 1.5 * EP * (DN4 - 2.) / (DN4 - 1.)
    I = (DNP + 0.1) * 0.5
    I = 2. * I
    if (I == N):
        A = A - 3. * EPS * (1. + 0.25 * EP) / (DN - 1.)
    R0 = REV * A**(-1./3.)
    for I in range(NG):
        XI = np.acos(X[I]) * DNP
        RI = R0 * (1. + EPS * np.cos(XI))
        R[I] = RI**2
        DR[I] = -R0 * EPS * DNP * np.sin(XI) / RI
    return R, DR



def main():
    # Входные данные, тестовые можно взять в description.txt
    AXI = 1   # Радиус эквивалентной сферы D
    RAT = 0.1   # Что-то непонятное. Описание как задается размер частицы
    LAM = np.acos(-1)*2  # Длина падающей волны
    MRR = 1.5  # Действительная часть показателя преломления
    MRI = 0.02  # Мнимая часть показателя преломления
    EPS = 0.5
    NP = -1
    DDELT = 0.001  # Точность вычислений
    NDGS = 2  # Какой-то параметр точности интегрирования

    P = np.acos(-1)
    NCHECK = 0

    if NP == -1 or NP == -2:
        NCKECK = 1

    if np.abs(RAT - 1) > 1e-8 and NP == -1:
        RAT = SAREA(EPS, RAT)
    if np.abs(RAT - 1) > 1e-8 and NP >= 0:
        RAT = SURFCH(NP, EPS, RAT)
    if np.abs(RAT - 1) > 1.e-8 and NP == -2:
        RAT = SAREAC(EPS, RAT)
    if NP == -3:
        RAT = DROP(RAT)

    print(f'RAT = {RAT}')

    if NP == -1 and EPS >= 1.:
        print(f'OBLITE SPHEROIDS, A/B = {EPS}')
    if NP == -1 and EPS < 1:
        print(f'PROLATE SPHEROIDS, A/B = {EPS}')
    if NP >= 0:
        print(f'CHEBYSHEV PERTICLES, T{NP}({EPS})')
    if NP == -2 and EPS >= 1.:
        print(f'OBLATE CYLINDERS, D/L = {EPS}')
    if NP == -2 and EPS < 1:
        print(f'PROLATE CYLINDERS, D/L = {EPS}')
    if NP == -3:
        print('Generalized chebyshev particles')

    print(f'ACCURACY OF COMPUTATIONS DDELT = {DDELT}')
    print(f'LAM = {LAM}, MRR = {MRR}, MRI = {MRI}')

    DDELT = 0.1 * DDELT

    if np.abs(RAT - 1) <= 1.e-6:
        print('EQUAL-VOLUME-SPHERE RADIUS = ', AXI)
    if np.abs(RAT - 1) > 1.e-6:
        print("EQUAL-SURFACE-AREA-SPHERE RADIUS = ", AXI)

    A = RAT * AXI
    XEV = 2 * P * A/LAM
    IXXX = XEV + 4.05 * XEV**(1./3.)
    INM1 = np.max(4, IXXX)
    if INM1 >= NPN1:
        print(f'CONVERGENCE IS NOT OBTAINED FOR NPN1 = {NPN1}.  EXECUTION TERMINATED')
        quit()
    QEXT1 = 0
    QSCA1 = 0
    for NMA in range(INM1-1, NPN1):
        NMAX = NMA
        MMAX = 0
        NGAUSS = NMAX * NDGS
        if NGAUSS > NPNG1:
            print('NGAUSS =', NGAUSS, ' I.E. IS GREATER THAN NPNG1. EXECUTION TERMINATED')
            quit()
        X, W, AN, ANN, S, SS = CONST(NGAUSS,NMAX,MMAX,P,X,W,AN,ANN,S,SS,NP,EPS)