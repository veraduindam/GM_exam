import numpy as np
import matplotlib.pyplot as plt
from functions import KnotVector, EvalBSpline


def CurveSplitting(Q, P_alt, Q_alt, p, V):
    # plotting the original control points
    plt.plot(P_alt[:,0], P_alt[:,1], 'o-', label='Control Points', color='k')
    plt.plot(Q_alt[:,0], Q_alt[:,1], 'o-', label='Control Points', color='k')

    # remove duplicate points
    Q_list = Q.tolist()
    n = len(Q_list)
    duplicate_items = []
    for index in range(1, n):
        if (Q_list[index - 1][0] == Q_list[index][0]) and (Q_list[index - 1][1] == Q_list[index][1]):
            duplicate_items.append(Q_list[index])

    for item in duplicate_items:
        Q_list.remove(item)

    Q = np.array(Q_list)

    Q_rolled = np.roll(Q, -(p - 1), axis=0)

    Q_alt = np.append(Q_rolled, [Q_rolled[0]], axis=0)

    n = len(Q_alt)
    # choosing the four new sets of control points and plotting open curves
    diff = p + 2
    sets = [Q_alt[0:diff], Q_alt[p + 1:p + 1 + diff], Q_alt[p + diff:p + diff + diff], Q_alt[p + diff + diff - 1:n + 1]]

    # reparametrize the first three on this knot vector
    for Q in sets:
        Q = np.array(Q, dtype="float64")
        UQ = KnotVector(Q, p, periodic=False)
        Z = np.array(EvalBSpline(V, Q, UQ, p, periodic=False)).T
        X, Y = Z.T
        plt.plot(X, Y)

    # settings of plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig("curve_splitting.png")
    plt.close()

    # returning the information to use in other files
    return sets

