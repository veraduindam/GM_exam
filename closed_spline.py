import numpy as np
import matplotlib.pyplot as plt
from functions import EvalBSpline, KnotVector


def ClosedSpline(P, p, V):
    # plotting the control points
    P_alt = np.append(P, [P[0]], axis=0)
    plt.plot(P_alt[:,0], P_alt[:,1], 'o-', label='Control Points', color='k')

    # calculating knot vector
    UP = KnotVector(P, p, periodic=True)

    # evaluating and plotting spline
    Z = np.array(EvalBSpline(V, P, UP, p, periodic=True)).T
    X, Y = Z.T
    plt.plot(X, Y, color='b')

    # settings of plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig("closed_spline.png")
    plt.close()

    # returning the information to use in other files
    return P_alt, UP

