import numpy as np
import matplotlib.pyplot as plt
from functions import EvalBSpline, CurveKnotIns


def KnotInsertion(P, P_alt, UP, p, V):
    # plotting the initial control points
    plt.plot(P_alt[:,0], P_alt[:,1], 'o-', label='Control Points', color='k')

    # repeatedly insert points and plot new control points
    points = [0, 0.3, 0.6, 0.9]

    for u in points:
        UQ, Q = CurveKnotIns(u, 0, p, UP, P, p)
        UP, P = UQ, Q

    Q_alt = np.append(Q, [Q[0]], axis=0)
    plt.plot(Q_alt[:,0], Q_alt[:,1], 'o-', label='Control Points', color='k')

    # plot the closed curve on these control points
    Q = np.array(Q, dtype="float64")
    Z = np.array(EvalBSpline(V, Q, UQ, p, periodic=True)).T
    X, Y = Z.T
    plt.plot(X, Y, color='b')

    # settings of plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig("knot_insertion.png")
    plt.close()

    # returning the information to use in other files
    return Q, Q_alt
