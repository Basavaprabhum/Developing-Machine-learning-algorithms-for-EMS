import numpy as np
import matplotlib.dates as mdates
import numpy as np
from scipy.interpolate import interp1d
import datetime as dt
def costElectricity(now,intervals,discretization):
    price = []
    tibber = [56.73, 44.56,39.35,34.67,38.36,38.91,43.81,88.46,90.85,48.33,35.58,35.13,37.88,38.47,37.1,35.95,36.04,53.93,59.74,55.63,49.9,48.01,39.39,26.55] #eur/MWh
    tibber = np.array(tibber)

    i = 0
    while i in range(intervals):
        price = np.append(price, tibber[now.hour])
        now += dt.timedelta(minutes = discretization)
        i += 1

    # for i in range(hours):
    #     price = np.append(price, float(tibber[i]) * np.ones(int(60/discretizationStep)))
    return price/1000
