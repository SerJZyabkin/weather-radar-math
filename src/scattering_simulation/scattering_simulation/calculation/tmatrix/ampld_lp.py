import numpy as np
from .ampld_par import *
from scipy.linalg.lapack import zgetrf, zgetri

class classDROP:
    C = [-0.0481, 0.0359, -0.1263, 0.0244, 0.0091, -0.0099, 0.0015, 0.0025, -0.0016, 0.0002, 0.001]
    R0V = 0

class classCBESS:
    def __init__(self):
        self.J = 0
        self.Y = 0
        self.JR = 0
        self.JI = 0
        self.DJ = 0
        self.DY = 0
        self.DJR = 0
        self.DJI = 0

class classT:
    def __init__(self):
        self.TR1 = np.zeros([NPN2, NPN2])
        self.TR2 = np.zeros([NPN2, NPN2])

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

##
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
        TA = np.max(TA, V)
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
    TB = np.max(TB, NMAX)
    NNMAX1 = 1.2 * np.sqrt(np.max(TA, NMAX)) + 3.
    NNMAX2 = (TB + 4. * (TB**(1./3.) + 1.2 * np.sqrt(TB)))
    NNMAX2 = NNMAX2 - NMAX + 5

    BESS(Z,ZR,ZI,NG,NMAX,NNMAX1,NNMAX2)
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

    CBESS.J = np.zeros([NPNG2, NPN1])
    CBESS.Y = np.zeros([NPNG2, NPN1])
    CBESS.JR = np.zeros([NPNG2, NPN1])
    CBESS.JI = np.zeros([NPNG2, NPN1])
    CBESS.DJ = np.zeros([NPNG2, NPN1])
    CBESS.DY = np.zeros([NPNG2, NPN1])
    CBESS.DJR = np.zeros([NPNG2, NPN1])
    CBESS.DJI = np.zeros([NPNG2, NPN1])

    for I in range(NG):
        XX = X(I)
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
    for I in range(L1):
        I1 = L - I
        Z[I1] = 1. / ((2.*I1+1.)*XX - Z[I1+1])

    Z0 = 1. /(XX - Z[0])
    Y0 = Z0 * np.cos(X) * XX
    Y1 = Y0 * Z[0]
    U[0] = Y0 - Y1*XX
    Y[0] = Y1
    for I in range(1, NMAX):
        YI1 = Y[I-1]
        YI = YI1 * Z[I]
        U[I] = YI1 - I*YI*XX
        Y[I] = YI
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
    for I in range(1,NMAX1):
        Y[I+1] = (2*I + 1)*X1*Y[I] - Y[I-1]

    V[0] =- X1 * (C+Y1)
    for I in range(1,NMAX):
        V[I] = Y[I-1] - I*X1*Y(I)
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
    for I in range(L1):
        I1 = L-I
        QF = 2*I1 + 1
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
    CI = np.sin(XR) * np.sinh(XI)

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

    for I in range(1, NMAX):
        QI = float(I) + 1
        CYI1R = CYR[I-1]
        CYI1I = CYI[I-1]
        CYIR = CYI1R*CZR[I] - CYI1I*CZI[I]
        CYII = CYI1I*CZR[I] + CYI1R*CZI[I]
        AR = CYIR*CXXR-CYII*CXXI
        AI = CYII*CXXR+CYIR*CXXI
        CUIR = CYI1R - QI*AR
        CUII = CYI1I - QI*AI
        CYR[I] = CYIR
        CYI[I] = CYII
        CUR[I] = CUIR
        CUI[I] = CUII
        YR[I] = CYIR
        YI[I] = CYII
        UR[I] = CUIR
        UI[I] = CUII

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

    for I in range(NGAUSS):
        I1 = NGAUSS + I
        I2 = NGAUSS - I - 1  # TODO
        DV1, DV2 = VIG(X[I1], NMAX, 0, DV1, DV2)
        for N in range(NMAX):
            SI = SIG[N]
            DD1 = DV1[N]
            DD2 = DV2[N]
            D1[I1, N] = DD1
            D2[I1, N] = DD2
            D1[I2, N] = DD1 * SI
            D2[I2, N] = -DD2 * SI

    for I in range(NGSS):
        RR[I] = W[I] * R[I]

    # for I in range(NGSS):
    #     WR = W[I] * R[I]
    #     DS[I] = S[I] * QM * WR
    #     DSS[I] = SS[I] * QMM
    #     RR[I] = WR
    for N1 in range(MM1 - 1, NMAX):
        AN1 = AN[N1]
        for N2 in range(MM1 - 1, NMAX):
            AN2 = AN[N2]
            AR12 = 0.
            AR21 = 0.
            AI12 = 0.
            AI21 = 0.
            GR12 = 0.
            GR21 = 0.
            GI12 = 0.
            GI21 = 0.
            if not (NCHECK == 1 and SIG[N1 + N2] < 0.):
                for I in range(NGSS):
                    D1N1 = D1[I, N1]
                    D2N1 = D2[I, N1]
                    D1N2 = D1[I, N2]
                    D2N2 = D2[I, N2]
                    A12 = D1N1 * D2N2
                    A21 = D2N1 * D1N2
                    A22 = D2N1 * D2N2
                    AA1 = A12 + A21

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

                    URI = DR[I]
                    RRI = RR[I]

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

            AN12 = ANN(N1, N2) * FACTOR
            TMAT99.R12[N1, N2] = AR12 * AN12
            TMAT99.R21[N1, N2] = AR21 * AN12
            TMAT99.I12[N1, N2] = AI12 * AN12
            TMAT99.I21[N1, N2] = AI21 * AN12
            TMAT99.RG12[N1, N2] = GR12 * AN12
            TMAT99.RG21[N1, N2] = GR21 * AN12
            TMAT99.IG12[N1, N2] = GI12 * AN12
            TMAT99.IG21[N1, N2] = GI21 * AN12

    TPIR = PIR
    TPII = PII
    TPPI = PPI
    NM = NMAX

    for N1 in range(MM1-1, NMAX):
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

    TT(NMAX, NCHECK)

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
        for N in range(NMAX):
            QN = float(N)
            QN1 = float(N+1)
            QN2 = float(2*N+1)
            D3 = (QN2*X*D2-QN*D1)/QN1
            DER = QS1*(QN1*QN/QN2)*(-D1+D3)
            DV1[N] = D2
            DV2[N] = DER
            D1 = D2
            D2 = D3
        return DV1, DV2

    QMM = float(M*M)
    for I in range(M):
        I2 = (I + 1) * 2
        A = A*np.sqrt((I2-1)/I2)*QS

    D1 = 0
    D2 = A
    for N in range(M - 1, NMAX):
        QN = float(N)
        QN2 = float(2*N+1)
        QN1 = float(N+1)
        QNM = np.sqrt(QN*QN-QMM)
        QNM1 = np.sqrt(QN1*QN1-QMM)
        D3 = (QN2*X*D2 - QNM*D1) / QNM1
        DER = QS1*(-QN1*QNM*D1 + QN*QNM1*D3) / QN2
        DV1[N] = D2
        DV2[N] = DER
        D1 = D2
        D2 = D3
    return DV1, DV2


def TT(NMAX,NCHECK):
    NDIM = NPN2
    NNMAX = 2 * NMAX
    ZQ = np.zeros([NPN2, NPN2], dtype=np.complex64)

    for I in range(NNMAX):
        for J in range(NNMAX):
            ZQ[I,J] = CTT.QR[I,J] + 1j * CTT.QI[I,J]

    ZQ, IPIV, INFO = zgetrf(ZQ, NPN2)  #TODO
    ZW, INFO = zgetri(ZQ, IPIV, NPN2)


def main():
    # Входные данные, тестовые можно взять в description.txt
    AXI = 1   # Радиус эквивалентной сферы D
    RAT = 0.1   # Что-то непонятное. Описание как задается размер частицы
    LAM = np.arccos(-1)*2  # Длина падающей волны
    MRR = 1.5  # Действительная часть показателя преломления
    MRI = 0.02  # Мнимая часть показателя преломления
    EPS = 0.5
    NP = -1
    DDELT = 0.001  # Точность вычислений
    NDGS = 2  # Какой-то параметр точности интегрирования

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

    TR1 = np.zeros([NPN2, NPN2])
    TI1 = np.zeros([NPN2, NPN2])

    PPI = 0
    PIR = 0
    PII = 0



    P = np.arccos(-1)
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

    INM1 = np.max([4., IXXX])
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
        PPI, PIR, PII, DDR, DRR, DRI = VARY(LAM, MRR, MRI, A, EPS, NP, NGAUSS, X, P, PPI,
                                            PIR, PII, R, DR, DDR, DRR, DRI, NMAX)

        TMATR0(NGAUSS, X, W, AN, ANN, S, SS, PPI, PIR, PII, R, DR, DDR, DRR, DRI, NMAX, NCHECK)
