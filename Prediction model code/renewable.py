
import datetime as dt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
def  renewableArray(realdays, discretizationStep):
    cname = "VarName3.txt"
    renewable = np.loadtxt(cname, dtype='float', delimiter=',')
    startingDate = dt.datetime(2023, 1, 1, 0)
    expiringDate = dt.datetime(2024, 1, 1, 0)
    days = mdates.drange(startingDate, expiringDate, dt.timedelta(hours=1))
    f = interp1d(days, renewable)
    renewableNew = f(realdays)
    return renewableNew/5

#Here you take into account the renewable production. You import from a txt file.