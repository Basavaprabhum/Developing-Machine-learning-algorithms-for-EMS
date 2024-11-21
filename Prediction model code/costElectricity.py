import numpy as np

import numpy as np
from scipy.interpolate import interp1d
import datetime as dt
def costElectricity(now,intervals,discretization):
    price = []
    tibber3 = [80.10,75.97,75.17,79.98,87.377,95.16,107.97,119.98,126.81,117.19,109.67,106.17,102.05,93.09,90.19,89.00,90.61,99.17,108.82,116.20,121.52,120.68,112.88,93.18] #eur/MWh 28/04
    tibber4 = [91.20,91.19,86.57,86.39,85.61,84.50,85.19,88.24,89.28,89.69,85.37,76.51,64.37,47.76,42.43,40.53,42.47,42.76,47.48,45.34,43.93,41.36,28.84,25.48] #eur/MWh 29/04
    tibber1 = [3.94, 3.97, 4.10, 4.01,4.39,6.23,13.35,76.92,88.18,80.54,57.31,59.95,20.07,17.24,16.08,14.50,19.99,70.54,71.34,74.68,73.57,45.90,4.64,3.14] #eur/MWh 22/05
    tibber2 = [2.84, 2.30, 1.70,1.70,2.40,3.91,9.95,71.46,74.34,70.07,60.29,44.99,57.42,20.10,5.72,4.33,12.81,53.94,61.27,66.88,67.47,68.62,19.91,2.64] #eur/MWh 23/05
    #tibberFinal = np.append(tibber1, tibber2)
    tibberFinal = np.append(tibber4, tibber3)
    #tibberFinal = np.append(tibberFinal,tibber1)
    tibberFinal = np.append(tibberFinal,tibber4)
    tibber = tibberFinal
    #tibber = [56.73, 44.56, 39.35, 34.67, 38.36, 38.91, 43.81, 88.46, 90.85, 48.33, 35.58, 35.13, 60.88, 62.47, 41.1,
              #35.95, 36.04, 53.93, 59.74, 55.63, 49.9, 48.01, 39.39, 26.55]
    #tibber = [56.73, 44.56, 39.35, 34.67, 38.36, 38.91, 43.81, 88.46, 90.85, 48.33, 35.58, 35.13, 37.88, 38.47, 37.1,
              #35.95, 36.04, 53.93, 59.74, 55.63, 49.9, 48.01, 39.39, 26.55]

    i = 0

    Price = []
    price = np.append(price, tibber[now.hour:])
    while i in range(int(intervals/4)):
        Price = np.append(Price, price[i] * np.ones(4))
        i += 1

    return Price/1000 #From here a price in  €/kWh



def guiPrice():
    tibber = [56.73, 44.56, 39.35, 34.67, 38.36, 38.91, 43.81, 88.46, 90.85, 48.33, 35.58, 35.13, 37.88, 38.47, 37.1,
              35.95, 36.04, 53.93, 59.74, 55.63, 49.9, 48.01, 39.39, 26.55]  # eur/MWh
    tibber = np.array(tibber)

    return np.max(tibber)/1000, np.min(tibber)/1000