# Schedify: Patient Scheduler Program

## Overview
Schedify is a patient scheduling program designed to optimize the arrangement of appointments using the PuLP library. The primary goal is to minimize the total workday (makespan) by efficiently scheduling various types and durations of appointments without overlap.

## Features
- **Optimization:** Utilizes Mixed Integer Programming to find optimal arrangements that minimize the total workday.
- **Flexibility:** Supports a variety of permutations for different patients, visit types, and durations.
- **Visualization:** Provides a graphical representation of the optimized schedule.

## Visual Representation
For example, if a naive scheduler might schedule 5 patients with varying visit types and durations, resulting in a more extended workday. In contrast, an optimized scheduler condenses the workday to its minimum, achieving efficiency.

### Naive Scheduler
![Naive Scheduler](https://cdn.mathpix.com/snip/images/Iw-XJinvIQy6KhtTamyERO845TnhXj-apdgZtuc9PxQ.original.fullsize.png)

### Optimized Scheduler
![Optimized Scheduler](https://cdn.mathpix.com/snip/images/ActAcRQbUPPXS5MtkIkK7p0n9QPoxOpOlJweEaXbau8.original.fullsize.png)

## Mathematical Model(Optimization Method)
The program employs Mixed Integer Programming to create a mathematical model, transforming the scheduling problem into Python code. The following components are crucial to the model:

### Input Data
Define the number of patients, their visit types, and appointment durations.
Here we are using our index page to take the input from the user using the web app
interface. The page look like this - 

![Input Page]()

We are taking this information and storing it in the data bases and showing it on the result page as - 

![Result Page]()

Next we are taking this data from the data base and storing it in the list format which can be seen below for the example purpose.

```python
# Define input data
patients = 5
visit_types = [0, 1, 2]  # Clinic Visit, Infusion, Nurse Follow-Up
durations = [[2, 3, 1], [1, 2, 2], [3, 1, 2], [2, 2, 1], [1, 3, 2]]
```

### Convenience Variables
Create variables like `valid_starts` and `jjm` to represent permutations of patient and visits, ensuring one patient is seen at a time.

```python
patients = len(clinic_sequence) # for knowing how many patient are there
vis = len(times[0])
valid_starts = [(j, m)
                for j in range(patients)
                for m in clinic_sequence[j]]
valid_starts.sort()

jjm = [(j, j_prime, m)
       for j in range(patients)
       for j_prime in range(patients)
       for m in range(vis)
       if j != j_prime
       and (j, m) in valid_starts
       and (j_prime, m) in valid_starts]
```

### Objective
Use PuLP to create a model called `schedule` that minimizes the day's makespan.

```python
# define the model to minimize the workday
schedule = LpProblem(name='Minimize_Schedule', sense=LpMinimize)
```

### Variables
Assign convenience variables (`x`, `y`, and `c`) as dictionaries in the scheduler model to store relevant permutations.

- `x` will be a dictionary of all valid starting values. It is always greater than or equal to 0, so the lower bound is set to 0. It is continuous.

- `y` will be a dictionary of `jjm`. Remember, this will operate to ensure that one patient goes at a time, rather than simultaneously. It's binary.

- `c` will represent the entire duration of the workday. The lower bound is set to 0, and it is continuous.

```python
# continuous x = the start time, for all the valid_starts
# x[j, m]
x = LpVariable.dicts("start_time", indexs=valid_starts, lowBound=0,
                     cat='Continuous')

# binary y = that patients j preceeds patient j_prime on visit m
# y[j ,j_prime, m]
y = LpVariable.dicts("precedence", indexs=jjm, cat='Binary')

# workday variable to capture the shortest makespan
c = LpVariable('makespan', lowBound=0, cat='Continuous')
```

### Constraints
Define constraints which can be given as - 
1) **Constraint 1:**

This ensures that a visit cannot begin until the last one ended. Mathematically, it looks like this:

![](https://cdn.mathpix.com/snip/images/hDhpTtDUHsxiRQPVlrdU-lVZ3gBiupcGdfZ00ynZ8lo.original.fullsize.png)

Where `curr_visit` represents the range of all visits for each patient (excluding the first visit because there is no prior visit), and `prior_visit` represents only the first prior visit from all visits for each patient.

In Python, we can add this logic to the schedule model like so:
```python
# visit sequence conntraint
for j in range(patients):   # For each patient
  for m_idx in range(1, len(clinic_sequence[j])): # for each visit sequence (except the first one)
    curr_visit = clinic_sequence[j][m_idx]
    prior_visit = clinic_sequence[j][m_idx - 1]

    # add constraint to the schedule
    schedule += x[j, curr_visit] >= x[j, prior_visit] + times[j][prior_visit]
```
2) **Constraint 2:**

The second constraint we’ll call a single-use constraint. Basically, this ensures that a patient is only in one visit, at one time. We can apply the big M constant to ensure the correctness of this constraint.

So if y=1, we know patient j_prime will precede patient j, and big M permits patient j_prime to come first. If y=0, then we know patient j precedes patient j_prime, and no big M applied will allow patient j to come first.

Mathematically this can be written as:

![](https://cdn.mathpix.com/snip/images/6hvDIAEWhilsD5Ef2RzuTG_5d-SyzV4ve9Szlc_q7Kk.original.fullsize.png)

In Python, this can be translated to:

```python
# single-use contraint

for j, j_prime, m in jjm:
  # if y = 1, we know patient j_prime preceeds patient j, and apply big M
  # if y = 0, we know patient j preceeds patient j_prime, and do no apply big M
  schedule += x[j, m] >= x[j_prime, m] + times[j_prime][m] - y[j, j_prime, m] * M
  schedule += x[j_prime, m] >= x[j, m] + times[j][m] - (1- y[j, j_prime, m]) * M
```
3) **Constraint 3:**

The final constraint ensures that the entire makespan is computed correctly. The makespan has to be at least as big as all patient visit start times plus their corresponding durations.

```python
# contraint to capture longest makespan

schedule += c   # put objective function into model

for j, m in valid_starts:
  schedule += c >= x[j, m] + times[j][m]
```

### Solve the Model
Call the `solve` function on the model to apply constraints and find an optimal schedule.

```python
status = schedule.solve()
```

### Show the Results
Print the results, displaying when each patient will start at each visit.
```python
print(f"status: {schedule.status}, {LpStatus[schedule.status]}")
print(f"Completion Time: {schedule.objective.value()}")

for j, m in valid_starts:
  if x[j, m].varValue >= 0:
    print(f"Patient {j + 1} starts in clinic {m} at time {x[j, m].varValue}")

```
## How to Use
1. Define input data for patients, visit types, and durations.
2. Run the program to obtain an optimized schedule.
3. Visualize the results and make adjustments if necessary.

