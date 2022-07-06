import numpy as np
import scipy.interpolate as si
import math


def KnotVector(P, p, periodic):
    """ Calculates the knot vector

        P:          array of control points
        p:          curve degree
        periodic:   True - curve is closed
                    False - curve is open
    """
    # calculate knot vector
    n = len(P)

    if periodic:
        U = np.arange(0 - p, n + p + p - 1) / (n - p)
    else:
        U = np.clip(np.arange(n + p + 1) - p, 0, n - p) / (n - p)

    return U


def EvalBSpline(u, P, U, p, periodic):
    """ Evaluate a B-spline in a point u and also return knot vector U

        u:          point in which to evaluate the B-spline (array-like)
        P:          array of control points
        U:          knot vector
        p:          curve degree
        periodic:   True - curve is closed
                    False - curve is open
    """
    # calculate knot vector
    P = np.asarray(P)
    n = len(P)

    # if periodic (closed), extend the point array to n + p + 1
    if periodic:
        factor, fraction = divmod(n + p + 1, n)
        P = np.concatenate((P,) * factor + (P[:fraction],))
        n = len(P)
        p = np.clip(p, 1, p)
    # if nonperiodic (open), prevent degree from exceeding n - 1
    else:
        p = np.clip(p, 1, n - 1)

    # calculate knot vector
    U = KnotVector(P, p, periodic)
 
    # calculate result (evaluate spline in point u)
    return si.splev(u, (U, P.T, p))


def IndexSearch(u, U):
    """ Find the index i such that u \in [u_k, u_{k+1})

        u:  evaluation point
        U:  knot vector
    """
    m = len(U)

    if u == U[m - 1]:
        k = m - 1
        return k

    low = 0
    high = m - 1
    mid = math.floor(high / 2)

    while (u < U[mid]) or (u >= U[mid + 1]):
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = math.floor((low + high) / 2)
    
    k = mid

    return k


def CurveKnotIns(u, s, r, UP, P, p):
    """ Perform knot insertion in a curve

        u:  knot to be inserted
        s:  multiplicity of knot
        r:  times knot should be inserted
        UP: knot vector associated with P
        P:  array of control points
        p:  curve degree
    """
    P = np.asarray(P)
    n = len(P)
    mp = len(UP)

    # find the index of knot to be inserted
    k = IndexSearch(u, UP)

    UQ = np.full(mp + r, None)
    # make new knot vector
    for i in range(k + 1):
        UQ[i] = UP[i]
    for i in range(1, r + 1):
        UQ[k + i] = u 
    for i in range(k + 1, mp):
        UQ[r + i] = UP[i]

    # generate new Q
    Q = np.full([n + r, 2], None)
    R = np.full([p + 1, 2], None)

    # save unaltered control points
    for i in range(k - p + 1):
        Q[i] = P[i]
    for i in range(k - s, n):
        Q[i + r] = P[i]

    # create temporary vector R
    for i in range(p - s + 1):
        R[i] = P[k - p + i]

    for j in range(1, r + 1):
        L = k - p + j

        for i in range(p - j - s + 1):
            alpha = (u - UP[L + i]) / (UP[i + k + 1] - UP[L + i])
            R[i] = alpha * R[i + 1] + (1 - alpha) * R[i]

        Q[L] = R[0]
        Q[k + r - j - s] = R[p - j - s]
    
    for i in range(L + 1, k - s + 1):
        Q[i] = R[i - L]

    return UQ, Q


def Surface(u, v, sets, U, p):
    """ Evaluates the surface S in the point (u, v)

        u:      point in [0, 1]
        v:      point in [0, 1]
        sets:   Q_1, Q_2, Q_3, Q_4
        U:      knot vector (same for every array of control points)
        p:      curve degree
    """
    Q_1 = sets[0]
    Q_2 = sets[2]
    Q_3 = sets[1]
    Q_4 = sets[3]
    
    c1_u = np.array(EvalBSpline(u, Q_1, U, p, periodic=False)).T
    c2_u = np.array(EvalBSpline(u, Q_2, U, p, periodic=False)).T
    r_c = np.multiply((1 - v), c1_u) + np.multiply(v, c2_u)

    d1_v = np.array(EvalBSpline(v, Q_3, U, p, periodic=False)).T
    d2_v = np.array(EvalBSpline(v, Q_4, U, p, periodic=False)).T
    r_d = np.multiply((1 - u), d1_v) + np.multiply(u, d2_v)

    c1_0 = np.array(EvalBSpline(0, Q_1, U, p, periodic=False)).T
    c1_1 = np.array(EvalBSpline(1, Q_1, U, p, periodic=False)).T
    c2_0 = np.array(EvalBSpline(0, Q_2, U, p, periodic=False)).T
    c2_1 = np.array(EvalBSpline(1, Q_2, U, p, periodic=False)).T
    r_cd = np.multiply(c1_0, (1 - u) * (1 - v)) + np.multiply(c1_1, u * (1 - v)) + np.multiply(c2_0, (1 - u) * v) 
    + np.multiply(c2_1, u * v)

    S = r_c + r_d - r_cd

    return S