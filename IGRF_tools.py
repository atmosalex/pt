import pyIGRF
from scipy import interpolate
import numpy as np
def arrange_IGRF_coeffs(year):
    f = interpolate.interp1d(pyIGRF.igrf.time, pyIGRF.igrf.coeffs, fill_value='extrapolate')
    coeffs = f(year)

    N = 13
    g = np.ones((N+1, N+1)) * np.nan
    h = np.ones((N+1, N+1)) * np.nan
    idx = 0
    for n in range(1, N + 1):
        # n, m=0
        m = 0
        g[n, m] = coeffs[idx]
        # print("g,{},{},{}".format(n,m,coeffs[idx]))
        idx += 1
        for m in range(1, n + 1):
            # n, m=1 to n-1
            g[n, m] = coeffs[idx]
            # print("g,{},{},{}".format(n,m,coeffs[idx]))
            idx += 1
            h[n, m] = coeffs[idx]
            # print("h,{},{},{}".format(n,m,coeffs[idx]))
            idx += 1

    return g, h