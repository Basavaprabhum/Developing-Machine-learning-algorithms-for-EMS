import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example datetime values
n_total = 10  # total number of datetime values
datetime_values = pd.date_range(start="2021-01-01", periods=n_total, freq='D')

# Function to simulate prediction algorithm
def predict(n_future):
    return np.random.rand(n_future)  # replace with actual prediction algorithm

# Number of cycles
n_cycles = 3

# Plotting
plt.figure(figsize=(10, 6))
for cycle in range(n_cycles):
    n_future = n_total - 2 * cycle
    predictions = predict(n_future)

    # Create an array with placeholders
    y_values = [None] * n_total
    y_values[2 * cycle:] = predictions

    # Plot
    plt.plot(datetime_values, y_values, label=f'Cycle {cycle + 1}')

plt.xlabel('Datetime')
plt.ylabel('Predictions')
plt.title('Prediction Values Over Cycles')
plt.xticks(rotation=45)
plt.legend()
plt.show()
