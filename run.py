import numpy as np
from closed_spline import ClosedSpline
from knot_insertion import KnotInsertion
from curve_splitting import CurveSplitting
from reparameterise import Reparameterise
from plot_surface import PlotSurface

P = np.array([[3, 1],
    [7, 1],
    [5.5, 2],
    [5.5, 6],
    [9, 9],
    [1, 9],
    [4.5, 6],
    [4.5, 2]])

p = 3
V = np.linspace(0, 1, num=100)

def Run(P, p):
    P_alt, UP = ClosedSpline(P, p, V)
    Q, Q_alt = KnotInsertion(P, P_alt, UP, p, V)
    sets = CurveSplitting(Q, P_alt, Q_alt, p, V)
    UQ = Reparameterise(sets, p, V)
    PlotSurface(sets, UQ, p, V)

Run(P, p)


