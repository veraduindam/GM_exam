from pickletools import read_decimalnl_long
import numpy as np
import matplotlib.pyplot as plt
from functions import EvalBSpline


def PlotSurface(sets, UQ, p, V):
    # plotting the surface edge
    for Q in sets:
        Q = np.array(Q, dtype="float64")
        Z = np.array(EvalBSpline(V, Q, UQ, p, periodic=False)).T
        X, Y = Z.T
        plt.plot(X, Y)

    # setting Q_1, Q_2, Q_3, Q_4
    Q_1 = sets[0]
    Q_2 = sets[2]
    Q_3 = sets[1]
    Q_4 = sets[3]

    # computing boundary vectors
    c1_0 = np.array(EvalBSpline(0, Q_1, UQ, p, periodic=False)).T
    c1_1 = np.array(EvalBSpline(1, Q_1, UQ, p, periodic=False)).T
    c2_0 = np.array(EvalBSpline(0, Q_2, UQ, p, periodic=False)).T
    c2_1 = np.array(EvalBSpline(1, Q_2, UQ, p, periodic=False)).T

    # finding all points in the surface
    points = []
    for u in np.linspace(0.01, 1, 100, endpoint=False):
        for v in np.linspace(0.01, 1, 100, endpoint=False):
            c1 = np.array(EvalBSpline(u, Q_1, UQ, p, periodic=False)).T
            c2 = np.array(EvalBSpline(u, Q_2, UQ, p, periodic=False)).T
            d1 = np.array(EvalBSpline(v, Q_3, UQ, p, periodic=False)).T
            d2 = np.array(EvalBSpline(v, Q_4, UQ, p, periodic=False)).T

            rc = np.multiply(c1, 1 - v) + np.multiply(c2, v)
            rd = np.multiply(d1, 1 - u) + np.multiply(d2, u)

            rcd = np.multiply(c1_0, (1 - u) * (1 - v)) + np.multiply(c1_1, u * (1 - v)) + np.multiply(c2_0, (1 - u) * v) 
            + np.multiply(c2_1, u * v)

            S = rc + rd - rcd

            points.append(S)

    # plotting all points
    Z = np.array(points)
    X, Y = Z.T
    plt.plot(X, Y)

    # settings of plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig("plot_surface.png")
    plt.close()
