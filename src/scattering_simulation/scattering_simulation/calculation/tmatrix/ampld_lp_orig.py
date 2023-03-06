import numpy as np
from .ampld_par import *
from scipy.linalg.lapack import zgetrf, zgetri
from copy import deepcopy

class classDROP:
    C = [-0.0481, 0.0359, -0.1263, 0.0244, 0.0091, -0.0099, 0.0015, 0.0025, -0.0016, 0.0002, 0.001]
    R0V = 0

class classTMAT:
    def __init__(self):
        self.RT11 = np.zeros([NPN6, NPN4, NPN4])
        self.RT12 = np.zeros([NPN6, NPN4, NPN4])
        self.RT21 = np.zeros([NPN6, NPN4, NPN4])
        self.RT22 = np.zeros([NPN6, NPN4, NPN4])
        self.IT11 = np.zeros([NPN6, NPN4, NPN4])
        self.IT12 = np.zeros([NPN6, NPN4, NPN4])
        self.IT21 = np.zeros([NPN6, NPN4, NPN4])
        self.IT22 = np.zeros([NPN6, NPN4, NPN4])


class classCBESS:
    def __init__(self):
        self.J = np.zeros([NPNG2, NPN1])
        self.Y = np.zeros([NPNG2, NPN1])
        self.JR = np.zeros([NPNG2, NPN1])
        self.JI = np.zeros([NPNG2, NPN1])
        self.DJ = np.zeros([NPNG2, NPN1])
        self.DY = np.zeros([NPNG2, NPN1])
        self.DJR = np.zeros([NPNG2, NPN1])
        self.DJI = np.zeros([NPNG2, NPN1])

class classT:
    def __init__(self):
        self.TR1 = np.zeros([NPN2, NPN2])
        self.TI1 = np.zeros([NPN2, NPN2])

class classTT:
    def __init__(self):
        self.QR = np.zeros([NPN2, NPN2])
        self.QI = np.zeros([NPN2, NPN2])
        self.RGQR = np.zeros([NPN2, NPN2])
        self.RGQI = np.zeros([NPN2, NPN2])

class classTMAT99:
    def __init__(self):
        self.R11 = np.zeros([NPN1, NPN1])
        self.R12 = np.zeros([NPN1, NPN1])
        self.R21 = np.zeros([NPN1, NPN1])
        self.R22 = np.zeros([NPN1, NPN1])

        self.I11 = np.zeros([NPN1, NPN1])
        self.I12 = np.zeros([NPN1, NPN1])
        self.I21 = np.zeros([NPN1, NPN1])
        self.I22 = np.zeros([NPN1, NPN1])

        self.RG11 = np.zeros([NPN1, NPN1])
        self.RG12 = np.zeros([NPN1, NPN1])
        self.RG21 = np.zeros([NPN1, NPN1])
        self.RG22 = np.zeros([NPN1, NPN1])

        self.IG11 = np.zeros([NPN1, NPN1])
        self.IG12 = np.zeros([NPN1, NPN1])
        self.IG21 = np.zeros([NPN1, NPN1])
        self.IG22 = np.zeros([NPN1, NPN1])

CBESS = classCBESS()
CDROP = classDROP()
CT = classT()
CTT = classTT()
TMAT99 = classTMAT99()
TMAT = classTMAT()

##
def GAUSS(N, IND1, IND2, Z, W):
    A = 1.
    B = 2.
    C = 3.
    IND = np.mod(N, 2)
    K = int (N / 2 + IND ) # ОКРУГЛЕНИЕ TODO
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
        R = 0.5 * (D**(2./3.) + D**(-1/3) * np.arcsin(E)/E)
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
        RI = 1 + classDROP.C[0]
        DRI = 0
        for N in range(1, NC + 1):
            XIN = XI * N
            RI = RI + classDROP.C[N] * np.cos(XIN)
            DRI = DRI - classDROP.C[N] * N * np.sin(XIN)
        SI = np.sin(XI)
        CI = X[I]
        RISI = RI * SI
        S = S + WI * RI * np.sqrt(RI**2 + DRI**2)
        V = V + WI * RI * RISI * (RISI - DRI * CI)

    RS = np.sqrt(S * 0.5)
    RV = (V * 3. / 4.)**(1./3.)

    if np.abs(RAT - 1.) > 1.e-8:
        RAT = RV / RS
    classDROP.R0V = 1 / RV
    print(f'r_0/r_ev = {classDROP.R0V}')
    for step in range(NC + 1):
        print(f'c_{step} = {classDROP.C[N]}')

    return RAT

def CONST(NGAUSS, NMAX, MMAX, P, X, W, AN, ANN, S, SS, NP, EPS):
    X1 = np.zeros(NPNG1)
    W1 = np.zeros(NPNG1)
    X2 = np.zeros(NPNG1)
    W2 = np.zeros(NPNG1)
    DD = np.zeros(NPN1)

    for N in range(1, NMAX + 1):
        NN = N * (N + 1)
        AN[N - 1] = NN
        D = np.sqrt((2 * N + 1) / NN )  # TODO 0 divinsion
        DD[N - 1] = D
        for N1 in range(N):
            DDD = D * DD[N1] * 0.5
            ANN[N - 1, N1] = DDD
            ANN[N1, N - 1] = DDD

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


def VARY(LAM, MRR, MRI, A, EPS, NP, NGAUSS, X, P, PPI, PIR, PII, R, DR, DDR, DRR, DRI, NMAX):
    Z = np.zeros(NPNG2)
    ZR = np.zeros(NPNG2)
    ZI = np.zeros(NPNG2)

    NG = NGAUSS * 2
    if NP > 0:
        R, DR = RSP2(X, NG, A, EPS, NP, R, DR)
    if NP == -1:
        R, DR = RSP1(X,NG,NGAUSS,A,EPS,NP,R,DR)
    if NP == -2:
        R, DR = RSP3(X,NG,NGAUSS,A,EPS,R,DR)
    if NP == -3:
        R, DR = RSP4(X,NG,A,R,DR)

    PI = P * 2./LAM
    PPI = PI*PI
    PIR = PPI*MRR
    PII = PPI*MRI
    V = 1./(MRR*MRR + MRI*MRI)
    PRR = MRR*V
    PRI = -MRI*V
    TA = 0.

    for I in range(NG):
        VV = np.sqrt(R[I])
        V = VV * PI
        TA = np.max([TA, V])
        VV = 1. / V
        DDR[I] = VV
        DRR[I] = PRR * VV
        DRI[I] = PRI * VV
        V1 = V * MRR
        V2 = V * MRI
        Z[I] = V
        ZR[I] = V1
        ZI[I] = V2

    if NMAX >= NPN1:
        print(f'NMAX = {NMAX}, i.e., grater than {NPN1}')
        quit()

    TB = TA * np.sqrt(MRR**2. + MRI**2.)
    TB = np.max([TB, NMAX])
    NNMAX1 = int(1.2 * np.sqrt(np.max([TA, NMAX]))) + 3
    NNMAX2 = int(TB + 4. * (TB**(1./3.) + 1.2 * np.sqrt(TB)))

    NNMAX2 = NNMAX2 - NMAX + 5

    BESS(Z,ZR,ZI,NG,NMAX, NNMAX1,NNMAX2)
    return PPI, PIR, PII, DDR, DRR, DRI

def RSP1(X,NG,NGAUSS,REV,EPS,NP,R,DR):
    A = REV*EPS**(1./3.)
    AA = A*A
    EE = EPS*EPS
    EE1 = EE-1.
    for I in range(NGAUSS):
        C = X[I]
        CC = C*C
        SS = 1. - CC
        S = np.sqrt(SS)
        RR = 1./(SS + EE*CC)
        R[I] = AA * RR
        R[NG-I-1]= R[I]
        DR[I]= RR*C*S*EE1
        DR[NG-I-1]= -DR[I]
    return R, DR


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
        XI = np.arccos(X[I]) * DNP
        RI = R0 * (1. + EPS * np.cos(XI))
        R[I] = RI**2
        DR[I] = -R0 * EPS * DNP * np.sin(XI) / RI
    return R, DR

def RSP3(X, NG, NGAUSS, REV, EPS, R, DR):
    H=REV * (2./(3.*EPS*EPS))**(1./3.)
    A = H*EPS
    for I in range(NGAUSS):
        CO = -X[I]
        SI = np.sqrt(1. - CO*CO)
        if SI/CO > A/H:
            RAD = A / SI
            RTHET = -A * CO / (SI * SI)
        else:
            RAD = H/CO
            RTHET = H*SI/(CO*CO)
        R[I] = RAD * RAD
        R[NG - I + 1] = R[I]
        DR[I] = -RTHET/RAD
        DR[NG - I - 1] = -DR[I]
    return R, DR


def RSP4(X, NG, REV, R, DR):
    NC = 10
    R0 = REV * classDROP.R0V
    for I in range(NG):
        XI = np.cos(X[I])
        RI = 1. + classDROP.C[0]
        DRI = 0.
        for N in range(NC):
            XIN = XI*N
            RI = RI + classDROP.C[N] * np.cos(XIN)
            DRI = DRI - classDROP.C[N] * N * np.sin(XIN)
        RI = RI*R0
        DRI = DRI*R0
        R[I] = RI*RI
        DR[I] = DRI/RI
    return R, DR

def BESS(X,XR,XI,NG,NMAX,NNMAX1,NNMAX2):
    AJ = np.zeros(NPN1)
    ADJ = np.zeros(NPN1)
    AJR = np.zeros(NPN1)
    AJI = np.zeros(NPN1)
    ADJR = np.zeros(NPN1)
    ADJI = np.zeros(NPN1)

    AY = np.zeros(NPN1)
    ADY = np.zeros(NPN1)

    CBESS.J = np.zeros([NPNG2, NPN1])
    CBESS.Y = np.zeros([NPNG2, NPN1])
    CBESS.JR = np.zeros([NPNG2, NPN1])
    CBESS.JI = np.zeros([NPNG2, NPN1])
    CBESS.DJ = np.zeros([NPNG2, NPN1])
    CBESS.DY = np.zeros([NPNG2, NPN1])
    CBESS.DJR = np.zeros([NPNG2, NPN1])
    CBESS.DJI = np.zeros([NPNG2, NPN1])

    for I in range(NG):
        XX = X[I]
        AJ, ADJ = RJB(XX, AJ, ADJ, NMAX, NNMAX1)
        AY, ADY = RYB(XX, AY, ADY, NMAX)

        YR = XR[I]
        YI = XI[I]
        AJR, AJI, ADJR, ADJI = CJB(YR, YI, AJR, AJI, ADJR, ADJI, NMAX, NNMAX2)
        for N in range(NMAX):
            CBESS.J[I,N] = AJ[N]
            CBESS.Y[I,N] = AY[N]
            CBESS.JR[I,N] = AJR[N]
            CBESS.JI[I,N] = AJI[N]
            CBESS.DJ[I,N] = ADJ[N]
            CBESS.DY[I,N] = ADY[N]
            CBESS.DJR[I,N] = ADJR[N]
            CBESS.DJI[I,N] = ADJI[N]


def RJB(X,Y,U,NMAX,NNMAX):
    Z = np.zeros(800)
    L = NMAX + NNMAX
    XX = 1. / X
    Z[L] = 1. /((2.* L + 1.) * XX)
    L1 = L - 1
    for I in range(1, L1 + 1):
        I1 = L - I
        Z[I1 - 1] = 1. / ((2.*I1+1.)*XX - Z[I1])

    Z0 = 1. /(XX - Z[0])
    Y0 = Z0 * np.cos(X) * XX
    Y1 = Y0 * Z[0]
    U[0] = Y0 - Y1*XX
    Y[0] = Y1
    for I in range(2, NMAX + 1):
        YI1 = Y[I - 2]
        YI = YI1 * Z[I - 1]
        U[I - 1] = YI1 - I*YI*XX
        Y[I - 1] = YI
    # print('CALL RJB', Y[0], Y[5], Y[NMAX - 1])
    return Y, U

def RYB(X, Y, V, NMAX):
    C = np.cos(X)
    S = np.sin(X)
    X1 = 1./X
    X2 = X1*X1
    X3 = X2*X1
    Y1 = -C*X2-S*X1
    Y[0] = Y1
    Y[1] =(-3.*X3 + X1) * C - 3.* X2 * S
    NMAX1 = NMAX-1
    for I in range(2, NMAX1 + 1):
        Y[I] = (2*I + 1)*X1*Y[I - 1] - Y[I-2]
    V[0] =- X1 * (C+Y1)
    for I in range(2, NMAX + 1):
        V[I - 1] = Y[I-2] - I*X1*Y[I - 1]
    return Y, V

def CJB(XR,XI,YR,YI,UR,UI,NMAX: int,NNMAX: int):
    CZR = np.zeros(1200)
    CZI = np.zeros(1200)
    CYR = np.zeros(NPN1)
    CYI = np.zeros(NPN1)
    CUR = np.zeros(NPN1)
    CUI = np.zeros(NPN1)

    L = NMAX + NNMAX
    XRXI = 1./(XR*XR+XI*XI)
    CXXR = XR*XRXI
    CXXI = -XI*XRXI
    QF = 1./(2*L+1)
    CZR[L] = XR*QF
    CZI[L] = XI*QF
    L1 = L-1
    for I in range(1, L1 + 1):
        I1 = L - I
        QF = 2 * I1 + 1
        AR = QF*CXXR - CZR[I1]
        AI = QF*CXXI - CZI[I1]
        ARI = 1. / (AR*AR+AI*AI)
        CZR[I1 - 1] = AR*ARI
        CZI[I1 - 1] = -AI*ARI
    AR = CXXR - CZR[0]
    AI = CXXI - CZI[0]
    ARI = 1. / (AR**2 + AI**2)
    CZ0R = AR * ARI
    CZ0I = -AI * ARI
    CR = np.cos(XR) * np.cosh(XI)
    CI = -np.sin(XR) * np.sinh(XI)

    AR = CZ0R * CR - CZ0I * CI
    AI = CZ0I * CR + CZ0R * CI
    CY0R = AR * CXXR - AI * CXXI
    CY0I = AI * CXXR + AR * CXXI
    CY1R = CY0R * CZR[0] - CY0I * CZI[0]
    CY1I = CY0I * CZR[0] + CY0R * CZI[0]
    AR = CY1R * CXXR - CY1I * CXXI
    AI = CY1I * CXXR + CY1R * CXXI
    CU1R = CY0R - AR
    CU1I = CY0I - AI

    CYR[0] = CY1R
    CYI[0] = CY1I
    CUR[0] = CU1R
    CUI[0] = CU1I
    YR[0] = CY1R
    YI[0] = CY1I
    UR[0] = CU1R
    UI[0] = CU1I

    for I in range(2, NMAX + 1):
        QI = float(I)
        CYI1R = CYR[I - 2]
        CYI1I = CYI[I - 2]
        CYIR = CYI1R*CZR[I - 1] - CYI1I*CZI[I - 1]
        CYII = CYI1I*CZR[I - 1] + CYI1R*CZI[I - 1]
        AR = CYIR*CXXR-CYII*CXXI
        AI = CYII*CXXR+CYIR*CXXI
        CUIR = CYI1R - QI*AR
        CUII = CYI1I - QI*AI
        CYR[I - 1] = CYIR
        CYI[I - 1] = CYII
        CUR[I - 1] = CUIR
        CUI[I - 1] = CUII
        YR[I - 1] = CYIR
        YI[I - 1] = CYII
        UR[I - 1] = CUIR
        UI[I - 1] = CUII

    return YR, YI, UR, UI

def TMATR0(NGAUSS, X, W, AN, ANN, S, SS, PPI, PIR, PII, R, DR, DDR, DRR, DRI, NMAX, NCHECK):
    SIG = np.zeros(NPN2)
    DV1 = np.zeros(NPN1)
    DV2 = np.zeros(NPN1)
    D1 = np.zeros([NPNG2, NPN1])
    D2 = np.zeros([NPNG2, NPN1])
    DS = np.zeros(NPNG2)
    DSS = np.zeros(NPNG2)
    RR = np.zeros(NPNG2)

    CTT.QR = np.zeros([NPN2, NPN2])
    CTT.QI = np.zeros([NPN2, NPN2])
    CTT.RGQR = np.zeros([NPN2, NPN2])
    CTT.RGQI = np.zeros([NPN2, NPN2])

    TQR = np.zeros([NPN2, NPN2])
    TQI = np.zeros([NPN2, NPN2])
    TRGQR = np.zeros([NPN2, NPN2])
    TRGQI = np.zeros([NPN2, NPN2])


    MM1 = 1
    NNMAX = NMAX + NMAX
    NG = 2 * NGAUSS
    NGSS = NG
    FACTOR = 1
    if NCHECK == 1:
        NGSS = NGAUSS
        FACTOR=2.
    SI = 1.
    for N in range(NNMAX):
        SI = -SI
        SIG[N] = SI

    for I in range(1, NGAUSS + 1):
        I1 = NGAUSS + I
        I2 = NGAUSS - I + 1
        DV1, DV2 = VIG(X[I1 - 1], NMAX, 0, DV1, DV2)
        for N in range(NMAX):
            SI = SIG[N]
            DD1 = DV1[N]
            DD2 = DV2[N]
            D1[I1 - 1, N] = DD1
            D2[I1 - 1, N] = DD2
            D1[I2 - 1 , N] = DD1 * SI
            D2[I2 - 1, N] = -DD2 * SI

    for I in range(NGSS):
        RR[I] = W[I] * R[I]

    for N1 in range(MM1, NMAX + 1):
        AN1 = AN[N1 - 1]
        for N2 in range(MM1, NMAX + 1):
            AN2 = AN[N2 - 1]
            AR12 = 0.
            AR21 = 0.
            AI12 = 0.
            AI21 = 0.
            GR12 = 0.
            GR21 = 0.
            GI12 = 0.
            GI21 = 0.
            if not (NCHECK == 1 and SIG[N1 + N2 - 1] < 0.):
                for I in range(1, NGSS + 1):
                    D1N1 = D1[I - 1, N1 - 1]
                    D2N1 = D2[I - 1, N1 - 1]
                    D1N2 = D1[I - 1, N2 - 1]
                    D2N2 = D2[I - 1, N2 - 1]
                    A12 = D1N1 * D2N2
                    A21 = D2N1 * D1N2
                    A22 = D2N1 * D2N2
                    AA1 = A12 + A21

                    QJ1 = CBESS.J[I - 1, N1 - 1]
                    QY1 = CBESS.Y[I - 1, N1 - 1]
                    QJR2 = CBESS.JR[I - 1, N2 - 1]
                    QJI2 = CBESS.JI[I - 1, N2 - 1]
                    QDJR2 = CBESS.DJR[I - 1, N2 - 1]
                    QDJI2 = CBESS.DJI[I - 1, N2 - 1]
                    QDJ1 = CBESS.DJ[I - 1, N1 - 1]
                    QDY1 = CBESS.DY[I - 1, N1 - 1]

                    C1R = QJR2 * QJ1
                    C1I = QJI2 * QJ1
                    B1R = C1R - QJI2 * QY1
                    B1I = C1I + QJR2 * QY1

                    C2R = QJR2 * QDJ1
                    C2I = QJI2 * QDJ1
                    B2R = C2R - QJI2 * QDY1
                    B2I = C2I + QJR2 * QDY1

                    DDRI = DDR[I - 1]
                    C3R = DDRI * C1R
                    C3I = DDRI * C1I
                    B3R = DDRI * B1R
                    B3I = DDRI * B1I

                    C4R = QDJR2 * QJ1
                    C4I = QDJI2 * QJ1
                    B4R = C4R - QDJI2 * QY1
                    B4I = C4I + QDJR2 * QY1

                    DRRI = DRR[I - 1]
                    DRII = DRI[I - 1]
                    C5R = C1R * DRRI - C1I * DRII
                    C5I = C1I * DRRI + C1R * DRII
                    B5R = B1R * DRRI - B1I * DRII
                    B5I = B1I * DRRI + B1R * DRII

                    URI = DR[I - 1]
                    RRI = RR[I - 1]

                    F1 = RRI * A22
                    F2 = RRI * URI * AN1 * A12
                    AR12 = AR12 + F1 * B2R + F2 * B3R
                    AI12 = AI12 + F1 * B2I + F2 * B3I
                    GR12 = GR12 + F1 * C2R + F2 * C3R
                    GI12 = GI12 + F1 * C2I + F2 * C3I

                    F2 = RRI * URI * AN2 * A21
                    AR21 = AR21 + F1 * B4R + F2 * B5R
                    AI21 = AI21 + F1 * B4I + F2 * B5I
                    GR21 = GR21 + F1 * C4R + F2 * C5R
                    GI21 = GI21 + F1 * C4I + F2 * C5I

            AN12 = ANN[N1 - 1, N2 - 1] * FACTOR
            TMAT99.R12[N1 - 1, N2 - 1] = AR12 * AN12
            TMAT99.R21[N1 - 1, N2 - 1] = AR21 * AN12
            TMAT99.I12[N1 - 1, N2 - 1] = AI12 * AN12
            TMAT99.I21[N1 - 1, N2 - 1] = AI21 * AN12
            TMAT99.RG12[N1 - 1, N2 - 1] = GR12 * AN12
            TMAT99.RG21[N1 - 1, N2 - 1] = GR21 * AN12
            TMAT99.IG12[N1 - 1, N2 - 1] = GI12 * AN12
            TMAT99.IG21[N1 - 1, N2 - 1] = GI21 * AN12

    TPIR = PIR
    TPII = PII
    TPPI = PPI
    NM = NMAX

    for N1 in range(MM1 - 1, NMAX):
        K1 = N1 - MM1 + 1
        KK1 = K1 + NM
        for N2 in range(MM1 - 1, NMAX):
            K2 = N2 - MM1 + 1
            KK2 = K2 + NM

            TAR12 = TMAT99.I12[N1, N2]
            TAI12 = -TMAT99.R12[N1, N2]
            TGR12 = TMAT99.IG12[N1, N2]
            TGI12 = -TMAT99.RG12[N1, N2]

            TAR21 = -TMAT99.I21[N1, N2]
            TAI21 = TMAT99.R21[N1, N2]
            TGR21 = -TMAT99.IG21[N1, N2]
            TGI21 = TMAT99.RG21[N1, N2]

            TQR[K1, K2] = TPIR * TAR21 - TPII * TAI21 + TPPI * TAR12
            TQI[K1, K2] = TPIR * TAI21 + TPII * TAR21 + TPPI * TAI12
            TRGQR[K1, K2] = TPIR * TGR21 - TPII * TGI21 + TPPI * TGR12
            TRGQI[K1, K2] = TPIR * TGI21 + TPII * TGR21 + TPPI * TGI12

            TQR[K1, KK2] = 0
            TQI[K1, KK2] = 0
            TRGQR[K1, KK2] = 0
            TRGQI[K1, KK2] = 0

            TQR[KK1, K2] = 0
            TQI[KK1, K2] = 0
            TRGQR[KK1, K2] = 0
            TRGQI[KK1, K2] = 0

            TQR[KK1, KK2] = TPIR * TAR12 - TPII * TAI12 + TPPI * TAR21
            TQI[KK1, KK2] = TPIR * TAI12 + TPII * TAR12 + TPPI * TAI21
            TRGQR[KK1, KK2] = TPIR * TGR12 - TPII * TGI12 + TPPI * TGR21
            TRGQI[KK1, KK2] = TPIR * TGI12 + TPII * TGR12 + TPPI * TGI21

    NNMAX = 2 * NM
    for N1 in range(NNMAX):
        for N2 in range(NNMAX):
            CTT.QR[N1, N2] = TQR[N1, N2]
            CTT.QI[N1, N2] = TQI[N1, N2]
            CTT.RGQR[N1, N2] = TRGQR[N1, N2]
            CTT.RGQI[N1, N2] = TRGQI[N1, N2]
    #        print('TMAT0', N1 + 1, N2 + 1, TQR[N1, N2], TQI[N1, N2], TRGQR[N1, N2], TRGQI[N1, N2])
    TT(NMAX, NCHECK)


def TMATR(M, NGAUSS, X, W, AN, ANN, S, SS, PPI, PIR, PII, R, DR, DDR, DRR, DRI, NMAX, NCHECK):
    SIG = np.zeros(NPN2)
    DV1 = np.zeros(NPN1)
    DV2 = np.zeros(NPN1)
    D1 = np.zeros([NPNG2, NPN1])
    D2 = np.zeros([NPNG2, NPN1])
    DS = np.zeros(NPNG2)
    DSS = np.zeros(NPNG2)
    RR = np.zeros(NPNG2)

    TQR = np.zeros([NPNG2, NPN2])
    TQI = np.zeros([NPNG2, NPN2])
    TRGQR = np.zeros([NPNG2, NPN2])
    TRGQI = np.zeros([NPNG2, NPN2])

    MM1 = M
    QM = float(M)
    QMM = QM * QM
    NG = 2*NGAUSS
    NGSS = NG
    FACTOR = 1.
    if NCHECK == 1:
        NGSS = NGAUSS
        FACTOR = 2.
    SI = 1
    NM = NMAX + NMAX
    for N in range(NM):
        SI = -SI
        SIG[N] = SI

    for I in range(NGSS):
        I1 = NGAUSS + I
        I2 = NGAUSS - I - 1

        DV1, DV2 = VIG(X[I1], NMAX, M, DV1, DV2)

        for N in range(NMAX):
            SI = SIG[N]
            DD1 = DV1[N]
            DD2 = DV2[N]
            D1[I1, N] = DD1
            D2[I1, N] = DD2
            D1[I2, N] = DD1 * SI
            D2[I2, N] = -DD2 * SI
         #   print(I1 + 1, I2 + 1, N + 1, DD1, DD2, DD1 * SI, -DD2*SI)

    for I in range(NGSS):
        WR = W[I] * R[I]
        DS[I] = S[I] * QM * WR
        DSS[I] = SS[I] * QMM
        RR[I] = WR
        # print('DSS', I + 1, WR, DS[I], DSS[I], QMM)


    for N1 in range(MM1-1, NMAX):
        AN1 = AN[N1]
        for N2 in range(MM1-1, NMAX):
            AN2 = AN[N2]
            AR11=0.
            AR12=0.
            AR21=0.
            AR22=0.
            AI11=0.
            AI12=0.
            AI21=0.
            AI22=0.
            GR11=0.
            GR12=0.
            GR21=0.
            GR22=0.
            GI11=0.
            GI12=0.
            GI21=0.
            GI22=0.
            SI=SIG[N1+N2 + 1]

            for I in range(NGSS):
                D1N1 = D1[I, N1]
                D2N1 = D2[I, N1]
                D1N2 = D1[I, N2]
                D2N2 = D2[I, N2]
                A11 = D1N1 * D1N2
                A12 = D1N1 * D2N2
                A21 = D2N1 * D1N2
                A22 = D2N1 * D2N2
                AA1 = A12 + A21
                AA2 = A11 * DSS[I] + A22
                QJ1 = CBESS.J[I, N1]
                QY1 = CBESS.Y[I, N1]
                QJR2 = CBESS.JR[I, N2]
                QJI2 = CBESS.JI[I, N2]
                QDJR2 = CBESS.DJR[I, N2]
                QDJI2 = CBESS.DJI[I, N2]
                QDJ1 = CBESS.DJ[I, N1]
                QDY1 = CBESS.DY[I, N1]

                C1R = QJR2 * QJ1
                C1I = QJI2 * QJ1
                B1R = C1R - QJI2 * QY1
                B1I = C1I + QJR2 * QY1

                C2R = QJR2 * QDJ1
                C2I = QJI2 * QDJ1
                B2R = C2R - QJI2 * QDY1
                B2I = C2I + QJR2 * QDY1

                DDRI = DDR[I]
                C3R = DDRI * C1R
                C3I = DDRI * C1I
                B3R = DDRI * B1R
                B3I = DDRI * B1I

                C4R = QDJR2 * QJ1
                C4I = QDJI2 * QJ1
                B4R = C4R - QDJI2 * QY1
                B4I = C4I + QDJR2 * QY1

                DRRI = DRR[I]
                DRII = DRI[I]
                C5R = C1R * DRRI - C1I * DRII
                C5I = C1I * DRRI + C1R * DRII
                B5R = B1R * DRRI - B1I * DRII
                B5I = B1I * DRRI + B1R * DRII

                C6R = QDJR2 * QDJ1
                C6I = QDJI2 * QDJ1
                B6R = C6R - QDJI2 * QDY1
                B6I = C6I + QDJR2 * QDY1

                C7R = C4R * DDRI
                C7I = C4I * DDRI
                B7R = B4R * DDRI
                B7I = B4I * DDRI

                C8R = C2R * DRRI - C2I * DRII
                C8I = C2I * DRRI + C2R * DRII
                B8R = B2R * DRRI - B2I * DRII
                B8I = B2I * DRRI + B2R * DRII

                URI = DR[I]
                DSI = DS[I]
                DSSI = DSS[I]
                RRI = RR[I]
                if NCHECK == 1 and SI > 0.:
                    F1 = RRI * AA2
                    F2 = RRI * URI * AN1 * A12
                    AR12 = AR12 + F1 * B2R + F2 * B3R
                    AI12 = AI12 + F1 * B2I + F2 * B3I
                    GR12 = GR12 + F1 * C2R + F2 * C3R
                    GI12 = GI12 + F1 * C2I + F2 * C3I

                    F2 = RRI * URI * AN2 * A21
                    AR21 = AR21 + F1 * B4R + F2 * B5R
                    AI21 = AI21 + F1 * B4I + F2 * B5I
                    GR21 = GR21 + F1 * C4R + F2 * C5R
                    GI21 = GI21 + F1 * C4I + F2 * C5I
                    continue

                else:
                    E1 = DSI*AA1
                    AR11 = AR11+E1*B1R
                    AI11 = AI11+E1*B1I
                    GR11 = GR11+E1*C1R
                    GI11 = GI11+E1*C1I

                    if NCHECK != 1:
                        F1 = RRI * AA2
                        F2 = RRI * URI * AN1 * A12
                        AR12 = AR12 + F1 * B2R + F2 * B3R
                        AI12 = AI12 + F1 * B2I + F2 * B3I
                        GR12 = GR12 + F1 * C2R + F2 * C3R
                        GI12 = GI12 + F1 * C2I + F2 * C3I

                        F2 = RRI * URI * AN2 * A21
                        AR21 = AR21 + F1 * B4R + F2 * B5R
                        AI21 = AI21 + F1 * B4I + F2 * B5I
                        GR21 = GR21 + F1 * C4R + F2 * C5R
                        GI21 = GI21 + F1 * C4I + F2 * C5I


                E2 = DSI * URI * A11
                E3 = E2 * AN2
                E2 = E2 * AN1
                AR22 = AR22 + E1 * B6R + E2 * B7R + E3 * B8R
                AI22 = AI22 + E1 * B6I + E2 * B7I + E3 * B8I
                GR22 = GR22 + E1 * C6R + E2 * C7R + E3 * C8R
                GI22 = GI22 + E1 * C6I + E2 * C7I + E3 * C8I

            AN12 = ANN[N1, N2] * FACTOR
            TMAT99.R11[N1, N2] = AR11 * AN12
            TMAT99.R12[N1, N2] = AR12 * AN12
            TMAT99.R21[N1, N2] = AR21 * AN12
            TMAT99.R22[N1, N2] = AR22 * AN12
            TMAT99.I11[N1, N2] = AI11 * AN12
            TMAT99.I12[N1, N2] = AI12 * AN12
            TMAT99.I21[N1, N2] = AI21 * AN12
            TMAT99.I22[N1, N2] = AI22 * AN12
            TMAT99.RG11[N1, N2] = GR11 * AN12
            TMAT99.RG12[N1, N2] = GR12 * AN12
            TMAT99.RG21[N1, N2] = GR21 * AN12
            TMAT99.RG22[N1, N2] = GR22 * AN12
            TMAT99.IG11[N1, N2] = GI11 * AN12
            TMAT99.IG12[N1, N2] = GI12 * AN12
            TMAT99.IG21[N1, N2] = GI21 * AN12
            TMAT99.IG22[N1, N2] = GI22 * AN12

    TPIR = PIR
    TPII = PII
    TPPI = PPI

    NM = NMAX - MM1 + 1

    for N1 in range(MM1 - 1, NMAX):
        K1 = N1 - MM1 + 1
        KK1 = K1 + NM
        for N2 in range(MM1 - 1, NMAX):
            K2 = N2 - MM1 + 1
            KK2 = K2 + NM

            TAR11 = -TMAT99.R11[N1, N2]
            TAI11 = -TMAT99.I11[N1, N2]
            TGR11 = -TMAT99.RG11[N1, N2]
            TGI11 = -TMAT99.IG11[N1, N2]
            #print('kkk1', TAR11, TAI11, TGR11, TGI11)
            TAR12 = TMAT99.I12[N1, N2]
            TAI12 = -TMAT99.R12[N1, N2]
            TGR12 = TMAT99.IG12[N1, N2]
            TGI12 = -TMAT99.RG12[N1, N2]

            TAR21 = -TMAT99.I21[N1, N2]
            TAI21 = TMAT99.R21[N1, N2]
            TGR21 = -TMAT99.IG21[N1, N2]
            TGI21 = TMAT99.RG21[N1, N2]

            TAR22 = -TMAT99.R22[N1, N2]
            TAI22 = -TMAT99.I22[N1, N2]
            TGR22 = -TMAT99.RG22[N1, N2]
            TGI22 = -TMAT99.IG22[N1, N2]

#            print('kkk1', TAR22, TAI22, TGR22, TGI22)

            TQR[K1, K2] = TPIR * TAR21 - TPII * TAI21 + TPPI * TAR12
            TQI[K1, K2] = TPIR * TAI21 + TPII * TAR21 + TPPI * TAI12
            TRGQR[K1, K2] = TPIR * TGR21 - TPII * TGI21 + TPPI * TGR12
            TRGQI[K1, K2] = TPIR * TGI21 + TPII * TGR21 + TPPI * TGI12

            TQR[K1, KK2] = TPIR * TAR11 - TPII * TAI11 + TPPI * TAR22
            TQI[K1, KK2] = TPIR * TAI11 + TPII * TAR11 + TPPI * TAI22
            TRGQR[K1, KK2] = TPIR * TGR11 - TPII * TGI11 + TPPI * TGR22
            TRGQI[K1, KK2] = TPIR * TGI11 + TPII * TGR11 + TPPI * TGI22

            TQR[KK1, K2] = TPIR * TAR22 - TPII * TAI22 + TPPI * TAR11
            TQI[KK1, K2] = TPIR * TAI22 + TPII * TAR22 + TPPI * TAI11
            TRGQR[KK1, K2] = TPIR * TGR22 - TPII * TGI22 + TPPI * TGR11
            TRGQI[KK1, K2] = TPIR * TGI22 + TPII * TGR22 + TPPI * TGI11

            TQR[KK1, KK2] = TPIR * TAR12 - TPII * TAI12 + TPPI * TAR21
            TQI[KK1, KK2] = TPIR * TAI12 + TPII * TAR12 + TPPI * TAI21
            TRGQR[KK1, KK2] = TPIR * TGR12 - TPII * TGI12 + TPPI * TGR21
            TRGQI[KK1, KK2] = TPIR * TGI12 + TPII * TGR12 + TPPI * TGI21

            # print('KKK', K1, K2, KK1, KK2)


    NNMAX = 2 * NM
    for N1 in range(NNMAX):
        for N2 in range(NNMAX):
            CTT.QR[N1, N2] = TQR[N1, N2]
            CTT.QI[N1, N2] = TQI[N1, N2]
            CTT.RGQR[N1, N2] = TRGQR[N1, N2]
            CTT.RGQI[N1, N2] = TRGQI[N1, N2]
#            print('TMAT', N1 + 1, N2 + 1, TQR[N1, N2], TQI[N1, N2], TRGQR[N1, N2], TRGQI[N1, N2])
    TT(NM, NCHECK)


def VIG(X, NMAX, M, DV1, DV2):
    A = 1
    QS = np.sqrt(1. - X*X)
    QS1 = 1./ QS

    for N in range(NMAX):
        DV1[N] = 0
        DV2[N] = 0

    if M == 0:
        D1 = 1
        D2 = X
        for N in range(1, NMAX + 1):
            QN = float(N)
            QN1 = float(N+1)
            QN2 = float(2*N+1)
            D3 = (QN2*X*D2-QN*D1)/QN1
            DER = QS1*(QN1*QN/QN2)*(-D1+D3)
            DV1[N - 1] = D2
            DV2[N - 1] = DER
            D1 = D2
            D2 = D3
        return DV1, DV2

    QMM = float(M*M)
    for I in range(1, M + 1):
        I2 = I * 2
        A = A*np.sqrt((I2-1)/I2)*QS

    D1 = 0
    D2 = A
    for N in range(M, NMAX + 1):
        QN = float(N)
        QN2 = float(2*N+1)
        QN1 = float(N+1)
        QNM = np.sqrt(QN*QN-QMM)
        QNM1 = np.sqrt(QN1*QN1-QMM)
        D3 = (QN2*X*D2 - QNM*D1) / QNM1
        DER = QS1*(-QN1*QNM*D1 + QN*QNM1*D3) / QN2
        DV1[N - 1] = D2
        DV2[N - 1] = DER
        D1 = D2
        D2 = D3
    return DV1, DV2

def TT(NMAX, NCHECK):
    NDIM = NPN2
    NNMAX = 2 * NMAX
    ZQ = np.zeros([NNMAX, NNMAX], dtype=np.complex64)

    for I in range(NNMAX):
        for J in range(NNMAX):
            ZQ[I,J] = CTT.QR[I,J] + 1j * CTT.QI[I,J]
   #         print('tt2', I + 1, J + 1, CTT.QR[I,J], CTT.QI[I,J])

  #  print('TT', ZQ[0, 0].real, ZQ[0, 0].imag, ZQ[NNMAX - 1, NNMAX - 1].real, ZQ[NNMAX - 1, NNMAX - 1].imag)

    ZQ, IPIV, INFO = zgetrf(ZQ, NPN2)  #TODO

  #  print('TT', ZQ[0, 0].real, ZQ[0, 0].imag, ZQ[NNMAX-1, NNMAX-1].real, ZQ[NNMAX-1, NNMAX-1].imag)

    ZQ, INFO = zgetri(ZQ, IPIV, NPN2)

    for I in range(NNMAX):
        for J in range(NNMAX):
            TR = 0
            TI = 0
            for K in range(NNMAX):
                ARR = CTT.RGQR[I, K]
                ARI = CTT.RGQI[I, K]
                AR = ZQ[K, J].real
                AI = ZQ[K, J].imag
                TR = TR - ARR * AR + ARI * AI
                TI = TI - ARR * AI - ARI * AR
            CT.TR1[I, J] = TR
            CT.TI1[I, J] = TI

#    print(CTT.RGQR[0, 0], CTT.RGQI[0, 0], CTT.RGQR[5, 5], CTT.RGQI[5, 5])
#    print(CT.TR1[0, 0], CT.TI1[0, 0], CT.TR1[5, 5], CT.TI1[5, 5])
#    print('TT', ZQ[0, 0].real, ZQ[0, 0].imag, ZQ[NNMAX-1, NNMAX-1].real, ZQ[NNMAX-1, NNMAX-1].imag)
#    print(IPIV[0], IPIV[3], IPIV[5], IPIV[7])

def AMPL(NMAX, DLAM, TL, TL1, PL, PL1, ALPHA, BETA):
    DV1 = np.zeros(NPN6)
    DV2 = np.zeros(NPN6)
    DV01 = np.zeros(NPN6)
    DV02 = np.zeros(NPN6)

    AL = np.zeros([3, 2])
    AL1 = np.zeros([3, 2])
    AP = np.zeros([2, 3])
    AP1 = np.zeros([2, 3])
    B = np.zeros([3, 3])
    R = np.zeros([2, 2])
    R1 = np.zeros([2, 2])
    C = np.zeros([3, 2])
    CAL = np.zeros([NPN4, NPN4], dtype=np.complex64)

    if (ALPHA < 0. or ALPHA > 360. or BETA < 0. or BETA > 180. or TL < 0. or TL > 180. or
        TL1 < 0. or TL1 > 180. or PL <0. or PL > 360. or PL1 < 0. or PL1 > 360.):
        print('AN ANGULAR PARAMETER IS OUTSIDE ITS ALLOWABLE RANGE')
        quit()

    PIN = np.arccos(-1.)
    PIN2 = PIN*0.5
    PI = PIN/180.
    ALPH = ALPHA * PI
    BET = BETA * PI
    THETL = TL * PI
    PHIL = PL * PI
    THETL1 = TL1 * PI
    PHIL1 = PL1 * PI
    EPS = 1.e-7

    if THETL < PIN2:
        THETL=THETL+EPS
    if THETL > PIN2:
        THETL=THETL-EPS
    if THETL1 < PIN2:
        THETL1=THETL1+EPS
    if THETL1 > PIN2:
        THETL1=THETL1-EPS
    if PHIL < PIN:
        PHIL=PHIL+EPS
    if PHIL > PIN:
        PHIL=PHIL-EPS
    if PHIL1 < PIN:
        PHIL1=PHIL1+EPS
    if PHIL1 > PIN:
        PHIL1=PHIL1-EPS
    if BET <= PIN2 and PIN2-BET <= EPS:
        BET=BET-EPS
    if BET > PIN2 and BET-PIN2 <= EPS:
        BET=BET+EPS

    # COMPUTE THETP, PHIP, THETP1, AND PHIP1, EQS. (8), (19), AND (20)
    CB = np.cos(BET)
    SB = np.sin(BET)
    CT = np.cos(THETL)
    ST = np.sin(THETL)
    CP = np.cos(PHIL - ALPH)
    SP = np.sin(PHIL - ALPH)
    CTP = CT * CB + ST * SB * CP
    THETP = np.arccos(CTP)
    CPP = CB * ST * CP - SB * CT
    SPP = ST * SP
    PHIP = np.arctan(SPP / CPP)

    if PHIP > 0. and SP < 0.:
        PHIP = PHIP + PIN
    if PHIP < 0. and SP > 0.:
        PHIP = PHIP + PIN
    if PHIP < 0.:
        PHIP = PHIP + 2.*PIN

    CT1 = np.cos(THETL1)
    ST1 = np.sin(THETL1)
    CP1 = np.cos(PHIL1 - ALPH)
    SP1 = np.sin(PHIL1 - ALPH)
    CTP1 = CT1 * CB + ST1 * SB * CP1
    THETP1 = np.arccos(CTP1)
    CPP1 = CB * ST1 * CP1 - SB * CT1
    SPP1 = ST1 * SP1
    PHIP1 = np.arctan(SPP1 / CPP1)
    if PHIP1 > 0. and SP1 < 0.:
        PHIP1 = PHIP1 + PIN
    if PHIP1 < 0. and SP1 > 0.:
        PHIP1 = PHIP1 + PIN
    if PHIP1 < 0.:
        PHIP1 = PHIP1 + 2.*PIN

    # C____________COMPUTE MATRIX BETA, EQ. (21)
    CA = np.cos(ALPH)
    SA = np.sin(ALPH)
    B[0, 0] = CA*CB
    B[0, 1] = SA*CB
    B[0, 2] = -SB
    B[1, 0] = -SA
    B[1, 1] = CA
    B[1, 2] = 0.
    B[2, 0] = CA*SB
    B[2, 1] = SA*SB
    B[2, 2] = CB

    # C____________COMPUTE MATRICES AL AND AL1, EQ. (14)
    CP = np.cos(PHIL)
    SP = np.sin(PHIL)
    CP1 = np.cos(PHIL1)
    SP1 = np.sin(PHIL1)
    AL[0, 0] = CT * CP
    AL[0, 1] = -SP
    AL[1, 0] = CT * SP
    AL[1, 1] = CP
    AL[2, 0] = -ST
    AL[2, 1] = 0.
    AL1[0, 0] = CT1 * CP1
    AL1[0, 1] = -SP1
    AL1[1, 0] = CT1 * SP1
    AL1[1, 1] = CP1
    AL1[2, 0] = -ST1
    AL1[2, 1] = 0.

    #C____________COMPUTE MATRICES AP^(-1) AND AP1^(-1), EQ. (15)

    CT = CTP
    ST = np.sin(THETP)
    CP = np.cos(PHIP)
    SP = np.sin(PHIP)
    CT1 = CTP1
    ST1 = np.sin(THETP1)
    CP1 = np.cos(PHIP1)
    SP1 = np.sin(PHIP1)
    AP[0, 0] = CT * CP
    AP[0, 1] = CT * SP
    AP[0, 2] = -ST
    AP[1, 0] = -SP
    AP[1, 1] = CP
    AP[1, 2] = 0.

    AP1[0, 0] = CT1 * CP1
    AP1[0, 1] = CT1 * SP1
    AP1[0, 2] = -ST1
    AP1[1, 0] = -SP1
    AP1[1, 1] = CP1
    AP1[1, 2] = 0.

    # C____________COMPUTE MATRICES R AND R^(-1), EQ. (13)

    for I in range(3):
        for J in range(2):
            X = 0.
            for K in range(3):
                X = X + B[I, K]*AL[K, J]
            C[I, J] = X

    for I in range(2):
        for J in range(2):
            X = 0.
            for K in range(3):
                X = X + AP[I,K]*C[K,J]
            R[I, J] = X

    for I in range(3):
        for J in range(2):
            X = 0.
            for K in range(3):
               X = X + B[I, K]*AL1[K, J]
            C[I, J] = X

    for I in range(2):
        for J in range(2):
            X = 0.
            for K in range(3):
               X = X + AP1[I, K]*C[K, J]
            R1[I,J] = X

    D = 1. / (R1[0, 0]*R1[1, 1] - R1[0, 1]*R1[1, 0])
    X = R1[0, 0]
    R1[0, 0] = R1[1, 1]*D
    R1[0, 1] = -R1[0, 1]*D
    R1[1, 0] = -R1[1, 0]*D
    R1[1, 1] = X*D

    CI = np.complex64(1j)

    for NN in range(1, NMAX + 1):
        for N in range(1, NMAX + 1):
            CN = CI**(NN - N - 1)
            DNN = float((2 * N + 1) * (2 * NN + 1))
            DNN = DNN / float(N * NN * (N + 1) * (NN + 1))
            RN = np.sqrt(DNN)
            CAL[N - 1, NN - 1] = CN * RN
         #   print("CAL", N, NN, CAL[N-1, NN-1].real, CAL[N-1, NN-1].imag)

    DCTH0 = CTP
    DCTH = CTP1
    PH = PHIP1 - PHIP

    VV = np.complex64(0.)
    VH = np.complex64(0.)
    HV = np.complex64(0.)
    HH = np.complex64(0.)

    for M in range(NMAX + 1):
    #for M in range(1):
        M1 = M + 1 - 1
        NMIN = np.max([M, 1])

        DV1, DV2 = VIGAMPL(DCTH, NMAX, M, DV1, DV2)
        DV01, DV02 = VIGAMPL(DCTH0, NMAX, M, DV01, DV02)

        FC = 2. * np.cos(M * PH)
        FS = 2. * np.sin(M * PH)

        for NN in range(NMIN - 1, NMAX):
            DV1NN = M * DV01[NN]
            DV2NN = DV02[NN]
            for N in range(NMIN - 1, NMAX):
              #  print('out1', NN + 1, N + 1, VV.real, VV.imag, HH.real, HH.imag)
                DV1N = M * DV1[N]
                DV2N = DV2[N]

                CT11 = TMAT.RT11[M1, N, NN] + 1j * TMAT.IT11[M1, N, NN]  # TR11 = RT11
                CT22 = TMAT.RT22[M1, N, NN] + 1j * TMAT.IT22[M1, N, NN]

                if M == 0:
                    CN = CAL[N, NN] * DV2N * DV2NN

                    VV = VV + CN * CT22
                    HH = HH + CN * CT11
                   # print('out1', NN + 1, N + 1, DV1N, DV2N, CN.real, CN.imag)
                   # print('out0', NN + 1, N + 1, TMAT.RT11[M1, N, NN], TMAT.IT11[M1, N, NN],
                   #         TMAT.RT22[M1, N, NN], TMAT.IT22[M1, N, NN] )
                   # print('out1', NN + 1, N + 1, VV.real, VV.imag, HH.real, HH.imag)
                else:
                    CT12 = TMAT.RT12[M1, N, NN] + 1j * TMAT.IT12[M1, N, NN]
                    CT21 = TMAT.RT21[M1, N, NN] + 1j * TMAT.IT21[M1, N, NN]

                    CN1 = CAL[N, NN] * FC
                    CN2 = CAL[N, NN] * FS

                    D11 = DV1N * DV1NN
                    D12 = DV1N * DV2NN
                    D21 = DV2N * DV1NN
                    D22 = DV2N * DV2NN

                    VV = VV + (CT11 * D11 + CT21 * D21 + CT12 * D12 + CT22 * D22) * CN1
                    VH = VH + (CT11 * D12 + CT21 * D22 + CT12 * D11 + CT22 * D21) * CN2
                    HV = HV - (CT11 * D21 + CT21 * D11 + CT12 * D22 + CT22 * D12) * CN2
                    HH = HH + (CT11 * D22 + CT21 * D12 + CT12 * D21 + CT22 * D11) * CN1

 #                   print('out2', NN + 1, N + 1, D11, D12, D21, D22)
             #   print('out2     ', VV.real, VV.imag, HH.real, HH.imag)
   #     quit()

    DK = 2. * PIN / DLAM
    VV = VV / DK
    VH = VH / DK
    HV = HV / DK
    HH = HH / DK

    CVV = VV * R[0, 0] + VH * R[1, 0]
    CVH = VV * R[0, 1] + VH * R[1, 1]
    CHV = HV * R[0, 0] + HH * R[1, 0]
    CHH = HV * R[0, 1] + HH * R[1, 1]

    VV = R1[0, 0] * CVV + R1[0, 1] * CHV
    VH = R1[0, 0] * CVH + R1[0, 1] * CHH
    HV = R1[1, 0] * CVV + R1[1, 1] * CHV
    HH = R1[1, 0] * CVH + R1[1, 1] * CHH

  #  print(f'thet0 = {TL}, thet = {TL1},  phi0 = {PL}, phi = {PL1}, alpha = {ALPHA}, beta = {BETA}')
  #   print('AMPLITUDE MATRIX')
  #   print(f'S11 = {VV.real} + i*{VV.imag}')
  #   print(f'S12 = {VH.real} + i*{VH.imag}')
  #   print(f'S21 = {HV.real} + i*{HV.imag}')
  #   print(f'S22 = {HH.real} + i*{HH.imag}')


    return VV, VH, HV, HH

def VIGAMPL (X, NMAX, M, DV1, DV2):
    """
C     Calculation of the functions
C     DV1(N)=dvig(0,m,n,arccos x)/sin(arccos x)
C     and
C     DV2(N)=[d/d(arccos x)] dvig(0,m,n,arccos x)
C     1.LE.N.LE.NMAX
C     0.LE.X.LE.1

    :param X:
    :param NMAX:
    :param M:
    :param DV1:
    :param DV2:
    :return:
    """
    for N in range(NMAX):
         DV1[N] = 0.
         DV2[N] = 0.

    DX = np.abs(X)
    if np.abs(1. - DX) >= 1.e-10:
        A = 1.
        QS = np.sqrt(1. - X * X)
        QS1 = 1. / QS
        DSI = QS1

        if M == 0:
            D1 = 1.
            D2 = X
            for N in range(1, NMAX + 1):
                QN = float(N)
                QN1 = float(N + 1)
                QN2 = float(2*N + 1)
                D3 = (QN2*X*D2 - QN*D1) / QN1
                DER = QS1*(QN1*QN / QN2)*(-D1+D3)
                DV1[N - 1] = D2*DSI
                DV2[N - 1] = DER
                D1 = D2
                D2 = D3
             #   print('STEP1', X, M, N, DV1[N - 1], DV2[N - 1])
            return DV1, DV2

        QMM = float(M * M)
        for I in range(M):
            I2 = (I + 1) * 2
            A = A * np.sqrt(float(I2 - 1) / float(I2)) * QS
        D1 = 0.
        D2 = A
        for N in range(M, NMAX + 1):
            QN = float(N)
            QN2 = float(2 * N + 1)
            QN1 = float(N + 1)
            QNM = np.sqrt(QN * QN - QMM)
            QNM1 = np.sqrt(QN1 * QN1 - QMM)
            D3 = (QN2 * X * D2 - QNM * D1) / QNM1
            DER = QS1 * (-QN1 * QNM * D1 + QN * QNM1 * D3) / QN2
            DV1[N - 1] = D2 * DSI
            DV2[N - 1] = DER
            D1 = D2
            D2 = D3
         #   print('STEP2', X, M, N + 1, DV1[N - 1], DV2[N - 1])
        return DV1, DV2

    if M != 1.:
        return DV1, DV2

    for N in range(1, NMAX + 1):
        DN = float(N * (N + 1))
        DN = 0.5 * np.sqrt(DN)
        if X < 0.:
            DN = DN * (-1) ** (N + 1)  # TODO ПОРЯДОК
        DV1[N - 1] = DN
        if X < 0.:
            DN = -DN
        DV2[N - 1] = DN
    #    print('STEP3', X, M, N + 1, DV1[N - 1], DV2[N - 1])
    return DV1, DV2


def main(wavelength, Deq: float, epsilon, gamma, alpha_tilt, beta_tilt, theta0, theta):

    # Входные данные, тестовые можно взять в description.txt
    AXI = 10   # Радиус эквивалентной сферы D
    RAT = 1   # Что-то непонятное. Описание как задается размер частицы
    LAM = np.arccos(-1)*2  # Длина падающей волны
    MRR = 1.5  # Действительная часть показателя преломления
    MRI = 0.02  # Мнимая часть показателя преломления
    EPS = 0.5
    NP = -1
    DDELT = 0.001  # Точность вычислений
    NDGS = 2  # Какой-то параметр точности интегрирования


    AXI = Deq
    MRR = epsilon.real
    MRI = epsilon.imag
    LAM = wavelength
    EPS = gamma

    X = np.zeros(NPNG2)
    W = np.zeros(NPNG2)
    S = np.zeros(NPNG2)
    SS = np.zeros(NPNG2)
    AN = np.zeros(NPN1)
    R = np.zeros(NPNG2)
    DR = np.zeros(NPNG2)
    DDR = np.zeros(NPNG2)
    DRR = np.zeros(NPNG2)
    DRI = np.zeros(NPNG2)
    ANN = np.zeros([NPN1, NPN1])

    PPI = 0
    PIR = 0
    PII = 0

    P = np.arccos(-1)
    NCHECK = 0

    if NP == -1 or NP == -2:
        NCHECK = 1

    if np.abs(RAT - 1) > 1e-8 and NP == -1:
        RAT = SAREA(EPS, RAT)
    if np.abs(RAT - 1) > 1e-8 and NP >= 0:
        RAT = SURFCH(NP, EPS, RAT)
    if np.abs(RAT - 1) > 1.e-8 and NP == -2:
        RAT = SAREAC(EPS, RAT)
    if NP == -3:
        RAT = DROP(RAT)

  #  print(f'RAT = {RAT}')
  #   if NP == -1 and EPS >= 1.:
  #       print(f'OBLITE SPHEROIDS, A/B = {EPS}')
  #   if NP == -1 and EPS < 1:
  #       print(f'PROLATE SPHEROIDS, A/B = {EPS}')
  #   if NP >= 0:
  #       print(f'CHEBYSHEV PERTICLES, T{NP}({EPS})')
  #   if NP == -2 and EPS >= 1.:
  #       print(f'OBLATE CYLINDERS, D/L = {EPS}')
  #   if NP == -2 and EPS < 1:
  #       print(f'PROLATE CYLINDERS, D/L = {EPS}')
  #   if NP == -3:
  #       print('Generalized chebyshev particles')
  #
  #   print(f'ACCURACY OF COMPUTATIONS DDELT = {DDELT}')
  #   print(f'LAM = {LAM}, MRR = {MRR}, MRI = {MRI}')

    DDELT = 0.1 * DDELT

    # if np.abs(RAT - 1) <= 1.e-6:
    #     print('EQUAL-VOLUME-SPHERE RADIUS = ', AXI)
    # if np.abs(RAT - 1) > 1.e-6:
    #     print("EQUAL-SURFACE-AREA-SPHERE RADIUS = ", AXI)

    A = RAT * AXI
    XEV = 2 * P * A/LAM
    IXXX = XEV + 4.05 * XEV**(1./3.)

    INM1 = int(np.max([4., IXXX]))  # Округление TODO
    if INM1 >= NPN1:
        print(f'CONVERGENCE IS NOT OBTAINED FOR NPN1 = {NPN1}.  EXECUTION TERMINATED')
        quit()
    QEXT1 = 0
    QSCA1 = 0
    for NMA in range(INM1, NPN1 + 1):
        NMAX = NMA
        MMAX = 1
        NGAUSS = NMAX * NDGS
        if NGAUSS > NPNG1:
            print('NGAUSS =', NGAUSS, ' I.E. IS GREATER THAN NPNG1. EXECUTION TERMINATED')
            quit()
        X, W, AN, ANN, S, SS = CONST(NGAUSS,NMAX,MMAX,P,X,W,AN,ANN,S,SS,NP,EPS)
        PPI, PIR, PII, DDR, DRR, DRI = VARY(LAM, MRR, MRI, A, EPS, NP, NGAUSS, X, P, PPI,
                                            PIR, PII, R, DR, DDR, DRR, DRI, NMAX)

        TMATR0(NGAUSS, X, W, AN, ANN, S, SS, PPI, PIR, PII, R, DR, DDR, DRR, DRI, NMAX, NCHECK)
        QEXT = 0
        QSCA = 0

        for N in range(1, NMAX + 1):
            N1 = N + NMAX - 1
            TR1NN = CT.TR1[N - 1, N - 1]
            TI1NN = CT.TI1[N - 1, N - 1]
            TR1NN1 = CT.TR1[N1, N1]
            TI1NN1 = CT.TI1[N1, N1]
            DN1 = 2 * N + 1
            QSCA = QSCA + DN1 * (TR1NN**2 + TI1NN**2 + TR1NN1**2 + TI1NN1**2)
            QEXT = QEXT + (TR1NN + TR1NN1) * DN1

        DSCA = np.abs((QSCA1 - QSCA) / QSCA)
        DEXT = np.abs((QEXT1 - QEXT) / QEXT)
        QEXT1 = QEXT
        QSCA1 = QSCA

        if DSCA <= DDELT and DEXT <= DDELT:
            break

        if NMA == NPN1:
            print("ERROR NMA == NPN")  # TODO может нужен NMA + 1 == NPN1
            quit()

    NNNGGG = NGAUSS + 1
    MMAX = NMAX
    if NGAUSS != NPNG1:
        for NGAUS in range(NNNGGG, NPNG1 + 1):
            NGAUSS = NGAUS
            NGGG = 2 * NGAUSS

            X, W, AN, ANN, S, SS = CONST(NGAUSS, NMAX, MMAX, P, X, W, AN, ANN, S, SS, NP, EPS)
            PPI, PIR, PII, DDR, DRR, DRI = VARY(LAM, MRR, MRI, A, EPS, NP, NGAUSS, X, P, PPI,
                                                PIR, PII, R, DR, DDR, DRR, DRI, NMAX)
            TMATR0(NGAUSS, X, W, AN, ANN, S, SS, PPI, PIR, PII, R, DR, DDR, DRR, DRI, NMAX, NCHECK)
            QEXT = 0
            QSCA = 0

            for N in range(1, NMAX + 1):
                N1 = N + NMAX - 1
                TR1NN = CT.TR1[N - 1, N - 1]
                TI1NN = CT.TI1[N - 1, N - 1]
                TR1NN1 = CT.TR1[N1, N1]
                TI1NN1 = CT.TI1[N1, N1]
                DN1 = 2 * N + 1
                QSCA = QSCA + DN1 * (TR1NN**2 + TI1NN**2 + TR1NN1**2 + TI1NN1**2)
                QEXT = QEXT + (TR1NN + TR1NN1) * DN1

            DSCA = np.abs((QSCA1 - QSCA) / QSCA)
            DEXT = np.abs((QEXT1 - QEXT) / QEXT)

            QEXT1 = QEXT
            QSCA1 = QSCA

            if DSCA <= DDELT and DEXT <= DDELT:
                break
            if NGAUS == NPNG1:
                print('WARNING NGAUS == NPNG1')

    QSCA = 0
    QEXT = 0
    NNM = NMAX * 2
    for N in range(NNM):
        QEXT = QEXT + CT.TR1[N, N]

    for N2 in range(NMAX):
        NN2 = N2 + NMAX
        for N1 in range(NMAX):
            NN1 = N1 + NMAX
            ZZ1 = CT.TR1[N1, N2]
            TMAT.RT11[0, N1, N2] = ZZ1
            ZZ2 = CT.TI1[N1, N2]
            TMAT.IT11[0, N1, N2] = ZZ2
            ZZ3 = CT.TR1[N1, NN2]
            TMAT.RT12[0, N1, N2] = ZZ3
            ZZ4 = CT.TI1[N1, NN2]
            TMAT.IT12[0, N1, N2] = ZZ4
            ZZ5 = CT.TR1[NN1, N2]
            TMAT.RT21[0,N1,N2] = ZZ5
            ZZ6 = CT.TI1[NN1, N2]
            TMAT.IT21[0,N1, N2] = ZZ6
            ZZ7 = CT.TR1[NN1, NN2]
            TMAT.RT22[0, N1, N2] = ZZ7
            ZZ8 = CT.TI1[NN1, NN2]
            TMAT.IT22[0, N1, N2] = ZZ8
            QSCA = QSCA + ZZ1 * ZZ1 + ZZ2 * ZZ2 + ZZ3 * ZZ3 + ZZ4 * ZZ4 + ZZ5 * ZZ5 + ZZ6 * ZZ6 + ZZ7 * ZZ7 + ZZ8 * ZZ8
 #   print(f' m = {0}, qxt = {np.abs(QEXT)},  qsc = {QSCA}, nmax = {NMAX}')
    for M in range(NMAX):#
#    for M in range(3):
        TMATR(M + 1, NGAUSS, X, W, AN, ANN, S, SS, PPI, PIR, PII, R, DR, DDR, DRR, DRI, NMAX, NCHECK)
        NM = NMAX - M
        M1 = M + 1
        QSC = 0
        for N2 in range(NM):
            NN2 = N2 + M
            N22 = N2 + NM
            for N1 in range(NM):
                NN1 = N1 + M
                N11 = N1 + NM
                ZZ1 = CT.TR1[N1, N2]
                TMAT.RT11[M1, NN1, NN2] = ZZ1
                ZZ2 = CT.TI1[N1, N2]
                TMAT.IT11[M1, NN1, NN2] = ZZ2
                ZZ3 = CT.TR1[N1, N22]
                TMAT.RT12[M1, NN1, NN2] = ZZ3
                ZZ4 = CT.TI1[N1, N22]
                TMAT.IT12[M1, NN1, NN2] = ZZ4
                ZZ5 = CT.TR1[N11, N2]
                TMAT.RT21[M1, NN1, NN2] = ZZ5
                ZZ6 = CT.TI1[N11, N2]
                TMAT.IT21[M1, NN1, NN2] = ZZ6
                ZZ7 = CT.TR1[N11, N22]
                TMAT.RT22[M1, NN1, NN2] = ZZ7
                ZZ8 = CT.TI1[N11, N22]
                TMAT.IT22[M1, NN1, NN2] = ZZ8
               # print('TMAT',M1 + 1, NN2+1, NN1+1, ZZ1, ZZ2, ZZ3, ZZ4)
                QSC = QSC + (ZZ1 * ZZ1 + ZZ2 * ZZ2 + ZZ3 * ZZ3 + ZZ4 * ZZ4 + ZZ5 * ZZ5 + ZZ6 * ZZ6 +
                             ZZ7 * ZZ7 + ZZ8 * ZZ8) * 2.
#                print(N1 + 1, N2 + 1, M1 + 1, NN1 + 1, NN2 + 1, N11 + 1, N22 + 1, QSC)
#                print(M + 1, N2 + 1, N1 + 1, QSC)

        NNM = 2 * NM
        QXT = 0

        for N in range(NNM):
            QXT = QXT + CT.TR1[N, N] * 2.

        QSCA = QSCA + QSC
        QEXT = QEXT + QXT

 #       print(f' m = {M + 1}, qxt = {np.abs(QXT)},  qsc = {QSC}, nmax = {NMAX}')
   # quit()
    WALB = - QSCA/QEXT
    if WALB > 1. + DDELT:
        print('WARNING: W IS GREATER THAN 1', WALB)

    # COMPUTATION OF THE AMPLITUDE AND PHASE MATRICES
    ALPHA = 145.
    BETA = 52.
    THET0 = 56.
    THET = 65.
    PHI0 = 114.
    PHI = 128.

    ALPHA = alpha_tilt
    BETA = beta_tilt
    THET0 = theta0
    THET = theta
    PHI0 = 0
    PHI = 0

    # AMPLITUDE MATRIX [Eqs. (2)-(4) of Ref. 6]
    S11, S12, S21, S22 = AMPL(NMAX, LAM, THET0, THET, PHI0, PHI, ALPHA, BETA)

    # PHASE MATRIX [Eqs. (13)-(29) of Ref. 6]
    Z11 = 0.5 * (S11 * np.conjugate(S11) + S12 * np.conjugate(S12) + S21 * np.conjugate(S21) + S22 * np.conjugate(S22))
    Z12 = 0.5 * (S11 * np.conjugate(S11) - S12 * np.conjugate(S12) + S21 * np.conjugate(S21) - S22 * np.conjugate(S22))
    Z13 = -S11 * np.conjugate(S12) - S22 * np.conjugate(S21)
    Z14 = (0. + 1.j) * (S11 * np.conjugate(S12) - S22 * np.conjugate(S21))

    Z21 = 0.5 * (S11 * np.conjugate(S11) + S12 * np.conjugate(S12) - S21 * np.conjugate(S21) - S22 * np.conjugate(S22))
    Z22 = 0.5 * (S11 * np.conjugate(S11) - S12 * np.conjugate(S12) - S21 * np.conjugate(S21) + S22 * np.conjugate(S22))
    Z23 = -S11 * np.conjugate(S12) + S22 * np.conjugate(S21)
    Z24 = (0. + 1.j) * (S11 * np.conjugate(S12) + S22 * np.conjugate(S21))

    Z31 = -S11 * np.conjugate(S21) - S22 * np.conjugate(S12)
    Z32 = -S11 * np.conjugate(S21) + S22 * np.conjugate(S12)
    Z33 = S11 * np.conjugate(S22) + S12 * np.conjugate(S21)
    Z34 = (0. + 1.j) * (S11 * np.conjugate(S22) + S21 * np.conjugate(S12))

    Z41 = (0. + 1.j) * (S21 * np.conjugate(S11) + S22 * np.conjugate(S12))
    Z42 = (0. + 1.j) * (S21 * np.conjugate(S11) - S22 * np.conjugate(S12))
    Z43 = (0. + 1.j) * (S22 * np.conjugate(S11) - S12 * np.conjugate(S21))
    Z44 = S22 * np.conjugate(S11) - S12 * np.conjugate(S21)

    # print('PHASE MATRIX')
    # print(f'{Z11}, {Z12}, {Z13}, {Z14}')
    # print(f'{Z21}, {Z22}, {Z23}, {Z24}')
    # print(f'{Z31}, {Z32}, {Z33}, {Z34}')
    # print(f'{Z41}, {Z42}, {Z43}, {Z44}')
    return S11, S12, S21, S22