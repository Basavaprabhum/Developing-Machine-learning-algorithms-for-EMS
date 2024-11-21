import matplotlib.dates as mdates
import datetime as dt
import csv
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from scipy.interpolate import interp1d
n = 5
def compute_averages(n, vector):
    num_averages = len(vector) // n
    output = np.zeros(num_averages)
    for i in range(num_averages):
        start = i * n
        end = (i + 1) * n
        output[i] = np.mean(vector[start:end:2])
    return output

def loadDefinitionnew(x):
    # Open the CSV file for reading
    with open('SumProfiles.Electricity.csv', newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile, delimiter=';')

        # Convert the rows in the CSV file to a list
        rows = [row for row in reader]

    # Convert the list to a NumPy array
    data = 60*np.array(rows,dtype = float)

    averaged = compute_averages(n, data)
    # Print the data array
    startingDate = dt.datetime(2023, 1, 1, 0, 0)
    expiringDate = dt.datetime(2024, 1, 1, 0, 0)
    days = mdates.drange(startingDate, expiringDate, dt.timedelta(minutes=n))
    f1 = interp1d(days, averaged)
    loadHome = f1(x)
    return loadHome
# startingDate = dt.datetime(startingDate.year, startingDate.month, startingDate.day, 0, 0, 0)
# expiringDate = startingDate + dt.timedelta(days=3)
#
#
#
# loadHomeForecast = f1(realdays)
# plt.plot(averaged)
#
# # Add a title and axis labels
# plt.title("My Plot")
# plt.xlabel("Index")
# plt.ylabel("Value")
#
# # Display the plot
# plt.show()