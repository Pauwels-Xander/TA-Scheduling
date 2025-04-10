import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpBinary, LpMaximize, lpSum, LpStatus, value

# Data
availability = np.random.choice([1, 0], size=(21, 9), p=[0.6, 0.4])
preference = np.random.choice([1, 0], size=(21, 9), p=[0.6, 0.4])


schedule = [
    [2],
    [6],
    [2],
    [4],
    [1],
    [8],
    [5],
    [4],
    [3]  # changed from 5 to 3
]

allocation = [
    [0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0]
]


# Create DataFrames
tas = [f"TA{i+1}" for i in range(len(availability))]
time_slots = [f"T{i+1}" for i in range(len(availability[0]))]

availability_df = pd.DataFrame(availability, index=tas, columns=time_slots)
preference_df   = pd.DataFrame(preference, index=tas, columns=time_slots)
schedule_df     = pd.DataFrame(schedule, index=time_slots, columns=["Number of groups"])
allocation_df   = pd.DataFrame(allocation, index=tas, columns=time_slots)

# Define the MILP model
model = LpProblem("TA_Scheduling", LpMaximize)

# Decision variables: x[j,i] = 1 if TA j is assigned to time slot i, 0 otherwise.
x = {(j, i): LpVariable(f"x_{j}_{i}", cat=LpBinary)
     for j in tas for i in time_slots}

# Objective: maximize total preference matches
model += lpSum(preference_df.loc[j, i] * x[(j, i)] for j in tas for i in time_slots)

# Constraints:
# 1) Each time slot must have the required number of TAs.
for i in time_slots:
    required = schedule_df.loc[i, "Number of groups"]
    model += lpSum(x[(j, i)] for j in tas) == required

# 2) A TA can only be assigned if available.
for j in tas:
    for i in time_slots:
        model += x[(j, i)] <= availability_df.loc[j, i]

# 3) Each TA can teach at most 2 tutorials.
for j in tas:
    model += lpSum(x[(j, i)] for i in time_slots) <= 2

# Solve the model
model.solve()

print("Status:", LpStatus[model.status])
print("Optimal Objective (Preference Matches):", value(model.objective))

# Display the optimized assignment.
result_df = pd.DataFrame(0, index=tas, columns=time_slots)
for j in tas:
    for i in time_slots:
        if x[(j, i)].varValue is not None and x[(j, i)].varValue > 0.5:
            result_df.loc[j, i] = 1
            print(f"{j} assigned to {i}")

print("\nHand-made allocation (for comparison):")
print(allocation_df)

# Save the result to an Excel file.
result_df.to_excel("optimized_allocation.xlsx")
