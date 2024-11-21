import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import os

def initialization(PopSize, D, LB, UB):
    SS_Boundary = len(LB) if isinstance(UB, (list, np.ndarray)) else 1
    if SS_Boundary == 1:
        Positions = np.random.rand(PopSize, D) * (UB - LB) + LB
    else:
        Positions = np.zeros((PopSize, D))
        for i in range(D):
            Positions[:, i] = np.random.rand(PopSize) * (UB[i] - LB[i]) + LB[i]
    return Positions

def GWO(PopSize, MaxT, LB, UB, D, Fobj):
    Alpha_Pos = np.zeros(D)
    Alpha_Fit = np.inf
    Beta_Pos = np.zeros(D)
    Beta_Fit = np.inf
    Delta_Pos = np.zeros(D)
    Delta_Fit = np.inf

    Positions = initialization(PopSize, D, UB, LB)
    Convergence_curve = np.zeros(MaxT)

    l = 0
    while l < MaxT:
        for i in range(Positions.shape[0]):
            BB_UB = Positions[i, :] > UB
            BB_LB = Positions[i, :] < LB
            Positions[i, :] = (Positions[i, :] * (~(BB_UB + BB_LB))) + UB * BB_UB + LB * BB_LB
            Fitness = Fobj(Positions[i, :])

            if Fitness < Alpha_Fit:
                Alpha_Fit = Fitness
                Alpha_Pos = Positions[i, :]

            if Fitness > Alpha_Fit and Fitness < Beta_Fit:
                Beta_Fit = Fitness
                Beta_Pos = Positions[i, :]

            if Fitness > Alpha_Fit and Fitness > Beta_Fit and Fitness < Delta_Fit:
                Delta_Fit = Fitness
                Delta_Pos = Positions[i, :]

        a = 2 - 1 * (2 / MaxT)
        for i in range(Positions.shape[0]):
            for j in range(Positions.shape[1]):
                r1 = np.random.random()
                r2 = np.random.random()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_Alpha = abs(C1 * Alpha_Pos[j] - Positions[i, j])
                X1 = Alpha_Pos[j] - A1 * D_Alpha

                r1 = np.random.random()
                r2 = np.random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_Beta = abs(C2 * Beta_Pos[j] - Positions[i, j])
                X2 = Beta_Pos[j] - A2 * D_Beta

                r1 = np.random.random()
                r2 = np.random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_Delta = abs(C3 * Delta_Pos[j] - Positions[i, j])
                X3 = Delta_Pos[j] - A3 * D_Delta

                Positions[i, j] = (X1 + X2 + X3) / 3
        l += 1
        Convergence_curve[l - 1] = Alpha_Fit
    return Alpha_Fit, Alpha_Pos, Convergence_curve


# Assuming you have X_train_norm, y_train defined
folder_path = r"C:\Users\Bmundas\OneDrive - Volvo Cars\Desktop\Data collected\New folder\Final_data_set"

# Specify the Excel file name
file_name = "cleaned_data.xlsx"

# Combine the folder path and file name
excel_file_path = os.path.join(folder_path, file_name)

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Assuming your data is stored in a DataFrame called df
# with columns like 'DateTime', 'PV Power', 'Temperature', 'Wind Speed', etc.

# Extract 'DateTime' column
datetime_col = df['DateTime']

# Extract columns to be scaled (excluding 'DateTime')
columns_to_scale = df.drop(['DateTime'], axis=1)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

scaled_columns= scale.fit_transform(columns_to_scale)
# Combine scaled columns and 'DateTime' back together
df_sc = pd.DataFrame(scaled_columns, columns=columns_to_scale.columns)
df_sc['DateTime'] = datetime_col

# Defining variables and splitting the data
y = df_sc['PV Power(kWh)']
x = df_sc.drop(['DateTime', 'PV Power(kWh)'], axis=1)

from sklearn.model_selection import train_test_split
X_training, X_testing, Y_training, Y_testing = train_test_split(x, y, test_size=0.01, random_state=0)
# Define the objective function for hyperparameter tuning
# Define the fitness function
def objective_function(solution):
    c, e, gamma = solution  # Parameters to be optimized
    regressor = SVR(kernel='rbf', C=c, epsilon=e, gamma=gamma)
    cv_scores = cross_val_score(regressor, X_training, Y_training, cv=5, scoring='neg_mean_squared_error')
    fitness = -np.mean(cv_scores)
    return fitness

# Define the optimization problem
hyperparameter_bounds = {
    "lb": [1, 0.001, 0.001],  # Lower bounds for C, epsilon, gamma
    "ub": [100, 10, 1],  # Upper bounds for C, epsilon, gamma
}
D = 3  # Dimensionality for SVR hyperparameters

# Configure GWO
epoch = 150
pop_size = 100
bounds = list(zip(hyperparameter_bounds["lb"], hyperparameter_bounds["ub"]))
problem = {
    "fit_func": objective_function,
    "lb": hyperparameter_bounds["lb"],
    "ub": hyperparameter_bounds["ub"],
    "minmax": "min",
}

print(bounds)
# Access each tuple element separately
lower_bound = [item[0] for item in bounds]
upper_bound = [item[1] for item in bounds]

# Run GWO to optimize hyperparameters
best_fitness, best_solution, convergence_curve = GWO(pop_size, epoch, lower_bound, upper_bound, len(lower_bound), objective_function)


# Display results
print("Best Fitness =", best_fitness)
print("Best Solution =", best_solution)

# Plot the convergence curve
plt.plot(convergence_curve)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('GWO Convergence Curve')
plt.show()
