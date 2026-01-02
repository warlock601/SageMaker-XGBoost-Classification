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

# to print out the starting data
cardio_df.head()

# similarly we can also print the last 4-5 rows by using tail()
cardio_df.tail()
```

- Perform Exploratory data analysis. cardio_df.isnull().sum() will give us non-zero numbers if we have any NULL elements. info() provides the total number of columns, not-null count for all the columns, data-type etc. 
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

- Let say we want to obtain the features of the individuals who are older than 64.8 years old.
```bash
age_greater_64 = cardio_df[cardio_df['age'>64.8]]
```


- Visualize Dataset.
  </br>
  Let say we want to plot the Histogram for all features (use 20 bins). Plot the correlation matrix and indicate if there exists any correlations between features.
  ```bash
  cardio_df.hist(bins=20, figsize=(15, 12))

  sns.pairplot(cardio_df)

  corr_matrix = cardio_df.corr()

  # Correlation matrix is best represented by a Heatmap-type representation
  sns.heatmap(corr_matrix, annot = True)

  ```

- Create Training and Testing dataset.
```bash
# split the dataframe into target and features

df_target = cardio_df['cardio']
df_final = cardio_df.drop(columns =['cardio'])

cardio_df.columns

df_final.shape

df_target.shape

```

```bash
#spliting the data in to test and train sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_final, df_target, test_size = 0.2)

# then we can check the shape of training and test dataset 
X_train.shape
y_train.shape
X_test.shape
X_test.shape
```


- Train & Test XG-Boost model in local mode (SageMaker built-in algorithms are not used here). We're not gonna use PCA algo in SageMaker here, instead we'll build quick short classifier by leveraging XG-Boost. And also       we'll learn how to do HyperParameter Optmization. On the instance itself we will install XG-Boost and use it in local mode.
```bash
# install xgboost

!pip install xgboost
```

We're using the default values of XGBClassifier(). We can change learning_rate, n_estimators etc.
```bash
# use xgboost model in local mode

# note that we have not performed any normalization or scaling since XGBoost is not sensitive to this.
# XGboost is a type of ensemble algorithms and works by selecting thresholds or cut points on features to split a node. 
# It doesn't really matter if the features are scaled or not.


from xgboost import XGBClassifier

# model = XGBClassifier(learning_rate=0.01, n_estimators=100, objective='binary:logistic')
model = XGBClassifier()

model.fit(X_train, y_train)
```
<img width="492" height="782" alt="image" src="https://github.com/user-attachments/assets/c8779232-dc63-4087-85ca-bdfc09d6a436" />

Here the classifier is trained with Base Score 0.5, max_depth of the tree is 6.
<img width="1117" height="175" alt="image" src="https://github.com/user-attachments/assets/f89afa92-c0a5-4b3a-ad6c-509e5613fb3c" />

```bash
# make predictions on test data

predict = model.predict(X_test)

predict
```


- Now we want to plot the confusion matrix for the training data and for the testing data. Also we want to plot the different scores for Training as well as for Testing separately. 

```bash
# Assess trained model performance on training dataset
predict_train = model.predict(X_train)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, predict_train)
plt.figure()
sns.heatmap(cm, annot=True)
```
<img width="453" height="277" alt="image" src="https://github.com/user-attachments/assets/a0566bfa-d02c-4a76-b697-65d2785a4742" />
