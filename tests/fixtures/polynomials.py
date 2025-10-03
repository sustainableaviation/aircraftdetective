# %%

import numpy as np
import pandas as pd

from aircraftdetective.utility.statistics import _compute_polynomials_from_dataframe

df = pd.read_csv("my.csv")

poly_overall = np.polynomial.Polynomial([145.93106807,50.85172733,-6.79756056,92.21929188], domain=[1958.46946229, 2019.76607775], window=[-1.,1.], symbol='x')
poly_aero = np.polynomial.Polynomial([16.92185143, 26.28760163, 16.41934499,0.41053852], domain=[1959.51077472, 2020.01806795], window=[-1.,1.], symbol='x')
poly_struct = np.polynomial.Polynomial([ 39.21925783,-6.79208613, -17.70301458,23.94885013], domain=[1958.06682578, 2020.07159905], window=[-1.,1.], symbol='x')
poly_eng = np.polynomial.Polynomial([ 52.66769313,1.48049973,11.88024979,42.61918829, -31.40911612], domain=[1958.06682578, 2020.07159905], window=[-1.,1.], symbol='x')

