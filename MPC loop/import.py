import numpy as np
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
import datetime
import csv
loadname = "load.txt"
bname = "matrixDay.txt"
cname = "VarName3.txt" #renewable

renewable = np.loadtxt(cname,dtype='float', delimiter = ',')
renewable = np.loadtxt(cname,dtype='float', delimiter = ',')
renewableMatrix = np.reshape(renewable, [365,24])
data = []
with open('load.txt', 'r') as file:
    data = [line.rstrip() for line in file.readlines()]

startingDate = dt.datetime(2016, 1, 1,0)
expiringDate = dt.datetime(2017, 1, 1,0)
days = mdates.drange(startingDate, expiringDate, dt.timedelta(minutes=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H %M'))
plt.gcf().autofmt_xdate()
plt.plot(days, data, "*--")

plt.title('Load prediction')
plt.xlabel('time [min]')
plt.ylabel('Power [kW]')
plt.legend('Load')
plt.show()


# f = interp1d(days, loadHome)
#
# now = datetime.datetime.now()
# minute = now.minute
# minute_rounded = 5 * round(minute/5)
# # create a new datetime object with the rounded minute
#
# now_rounded = now.replace(minute=minute_rounded)
# final = now_rounded + dt.timedelta(hours=15,minutes=15)
#
# realdays = mdates.drange(now_rounded, final , dt.timedelta(minutes=5))
#
# loadnew= f(realdays)
#
#
#
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H %M'))
# plt.gcf().autofmt_xdate()
# plt.plot(days, loadHome, "*--")
# plt.plot(realdays, loadnew, "o--")
# plt.title('Load prediction')
# plt.xlabel('time [min]')
# plt.ylabel('Power [kW]')
# plt.legend('Load')
# plt.show()