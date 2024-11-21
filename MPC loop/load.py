import datetime as dt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
def  loadArray(realdays,now,accuracy):
    #loadname = "normLoadMinutes.txt"
    loadname = "loadprogen.csv"
    # with open('load.txt', 'r') as file:
    #     loadHome = [line.rstrip() for line in file.readlines()]
    loadHome = np.loadtxt(loadname, dtype='float', delimiter=';')
    loadHomeForecast = []
    loadHomeForecast =np.append(np.append( np.append(loadHomeForecast, loadHome[0,:]), loadHome[1,:]),loadHome[2, :])/1000
    loadHomeReal = []
    loadHomeReal = np.append(np.append(np.append(loadHomeReal, loadHome[3, :]), loadHome[4, :]),loadHome[5, :]) / 1000
    difference = (loadHomeForecast - loadHomeReal)
    loadHomeForecast = loadHomeReal + difference * accuracy
    startingDate = now
    startingDate = dt.datetime(startingDate.year, startingDate.month, startingDate.day, 0, 0, 0)
    expiringDate = startingDate + dt.timedelta(days=3)
    days = mdates.drange(startingDate, expiringDate, dt.timedelta(minutes=15)) #qui e 15 minuti perche il carico viene generato con una frequenza di 15 minuti

    f1 = interp1d(days, loadHomeForecast)  #real day e´un vettore discretizzato come vuoi tu! sulla base di discretization step
    loadHomeForecast = f1(realdays)

    f2 = interp1d(days, loadHomeReal)
    loadHomeReal = f2(realdays)

    return loadHomeForecast , loadHomeReal

#In questo momemtno il carico e´il mio per il progetto di solar. Commentando irga 6 9 e la normale a riga 15 e togliendo
# i primi due trovi il carico del tedesco . Carico del poli invece ancora nulla.