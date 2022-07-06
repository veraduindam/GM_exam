import numpy as np
import matplotlib.pyplot as plt
from functions import KnotVector, EvalBSpline

def Reparameterise(sets, p, V):
    # find all the inner points in the knot vectors and create union knot vector
    inner_points = []
    for Q in sets:
        UQ = KnotVector(Q, p, periodic=False)
        m = len(UQ)
        for point in UQ[p + 1:m - p - 1]:
            inner_points.append(point)

    # create new uniform knot vector
    inner_points = set(inner_points)
    inner_points = list(inner_points)
    inner_points.sort()

    UQ = np.array([0] * (p + 1) + inner_points + [1] * (p + 1))

    for Q in sets:
        Q = np.array(Q, dtype="float64")
        Z = np.array(EvalBSpline(V, Q, UQ, p, periodic=False)).T
        X, Y = Z.T
        plt.plot(X, Y)

    # settings of plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig("reparameterise.png")
    plt.close()

    # returning the information to use in other files
    return UQ


