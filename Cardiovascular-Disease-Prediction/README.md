# Cardiovascular Disease Prediction

- Start by uploading the required files such as the dataset. Selected the kernel as "python3 (ipykernel).

- Aim of the problem is to detect the presence or absence of cardiovascular disease in person based on the given features. Features available are:
```bash
Age | Objective Feature | age | int (days)
Height | Objective Feature | height | int (cm) |
Weight | Objective Feature | weight | float (kg) |
Gender | Objective Feature | gender | categorical code |
Systolic blood pressure | Examination Feature | ap_hi | int |
Diastolic blood pressure | Examination Feature | ap_lo | int |
Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
Smoking | Subjective Feature | smoke | binary |
Alcohol intake | Subjective Feature | alco | binary |
Physical activity | Subjective Feature | active | binary |
Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

```
Note that:
  Objective: factual information; </br>
  Examination: results of medical examination; </br>
  Subjective: information given by the patient. </br>



- Import LIBRARIES & DATASETS
```bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# read the csv file 
cardio_df = pd.read_csv("cardio_train.csv", sep=";")

# to print out the data
cardio_df.head()
```

- Perform Exploratory data analysis.
```bash
# Drop id
cardio_df = cardio_df.drop(columns = 'id')

# since the age is given in days, we convert it into years
cardio_df['age'] = cardio_df['age']/365

cardio_df.head()

# checking the null values
cardio_df.isnull().sum()

# Checking the dataframe information
cardio_df.info()

# Statistical summary of the dataframe
cardio_df.describe()


```
