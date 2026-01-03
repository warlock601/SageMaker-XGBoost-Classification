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
Here we can see that the model is doing not that great. In the Confusion matrix there are a lot of False Positives and False Negatives.
<img width="453" height="277" alt="image" src="https://github.com/user-attachments/assets/a0566bfa-d02c-4a76-b697-65d2785a4742" />

- Then we print the metrics for training and testing dataset. These things will be imported: precision_score, recall_score, accuracy_score.
```bash
# print metrics for training dataset

from sklearn.metrics import precision_score, recall_score, accuracy_score

print("Precision = {}".format(precision_score(y_train, predict_train)))
print("Recall = {}".format(recall_score(y_train, predict_train)))
print("Accuracy = {}".format(accuracy_score(y_train, predict_train)))
```
```bash
# print metrics for testing dataset

print("Precision = {}".format(precision_score(y_test, predict)))
print("Recall = {}".format(recall_score(y_test, predict)))
print("Accuracy = {}".format(accuracy_score(y_test, predict)))
```

- Generate the Confusion Matrix for testing data.
```bash
# plot the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predict)
plt.figure()
sns.heatmap(cm, annot=True)
```
<img width="453" height="277" alt="image" src="https://github.com/user-attachments/assets/df8967d8-cd19-41d3-928d-05e5cd360354" />

We can play with this by increasing or decreasing the number of n_estimators ans max_depth. </br>
</br>
</br>
n_estimators & max_depth
- n_estimators: This hyperparameter Determines the number of decision trees in the ensemble model. Each tree learns from the residuals of the previous ones. A higher number of estimators generally improves
  performance up to a point, but can also lead to overfitting and increased training time if too high.
- max_depth: This parameter limits how deep each individual decision tree can grow. A deeper tree can model more complex relationships in the data, but it also increases the risk of overfitting the training data.            Conversely, a shallow tree might underfit the data. The optimal value typically lies between 3 and 10 and is found through cross-validation or hyperparameter tuning



- Let say we want to increase the n_estimators to 500 and max_depth to 20. Then we retrain the model using this. 
```bash

from xgboost import XGBClassifier
model = XGBClassifier(learning_rate = 0.01, n_estimators = 500, max_depth = 20)
model.fit(X_train, y_train)

```



- Perform Dimensionality Reduction using PCA (SageMaker buildt-in Alogrithm)
```bash
# Boto3 is the Amazon Web Services (AWS) Software Development Kit (SDK) for Python
# Boto3 allows Python developer to write software that makes use of services like Amazon S3 and Amazon EC2



import sagemaker
import boto3
from sagemaker import Session

# Let's create a Sagemaker session
sagemaker_session = sagemaker.Session()
bucket = Session().default_bucket() 
prefix = 'pca'  # prefix is the subfolder within the bucket.

#Let's get the execution role for the notebook instance. 
# This is the IAM role that you created when you created your notebook instance. You pass the role to the training job.
# Note that AWS Identity and Access Management (IAM) role that Amazon SageMaker can assume to perform tasks on your behalf (for example, reading training results, called model artifacts, from the S3 bucket and writing training results to Amazon S3). 

role = sagemaker.get_execution_role()
```

Then we convert the Numpy array into RecordIO format. "smac" is sagemaker common library in SageMaker which is primarily a collection of preconfigured Python libraries and tools designed to streamline and automate common, labor-intensive machine learning (ML) workflows. Use cases: Model building & training, hyperparameter tuning & optimization, data prep etc. smac can also be used to do data conversion, here we're using it for that only.
```bash
import io # The io module allows for dealing with various types of I/O (text I/O, binary I/O and raw I/O). 
import numpy as np
import sagemaker.amazon.common as smac # sagemaker common libary

# Code below converts the data in numpy array format to RecordIO format
# This is the format required by Sagemaker PCA

buf = io.BytesIO() # create an in-memory byte array (buf is a buffer I will be writing to)
df_matrix = df_final.to_numpy() # convert the dataframe into 2-dimensional array
smac.write_numpy_to_dense_tensor(buf, df_matrix)
buf.seek(0)

# When you write to in-memory byte arrays, it increments 1 every time you write to it
# Let's reset that back to zero
```
We don't have any targets here, like there are no predictions that we're trying to do. All we're trying to do is to take the inputs and perform an unsupervised learning strategy to reduce the number of features and generate componenets for us. So there is no output here. We're gonna use the outputs later on, when we train the XGBoost algorithm on the reduced componenets. 

Then we create a key (which will be name of the file). Then we upload the data in record-io format to S3 bucket. 
```bash
import os

# Code to upload RecordIO data to S3
 
# Key refers to the name of the file 
 
key = 'pca'

#following code uploads the data in record-io format to S3 bucket to be accessed later for training
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)

# Let's print out the training data location in s3
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)


print('uploaded training data location: {}'.format(s3_train_data))
```
<img width="1676" height="422" alt="image" src="https://github.com/user-attachments/assets/b8cd89f4-e9c4-41b8-b7a2-70770a98529a" />


```bash
# create output placeholder in S3 bucket to store the PCA output

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))
```

Now we need to obtain a reference to PCA container image exactly the same as we have done before.
```bash
# This code is used to get the training container of sagemaker built-in algorithms
# all we have to do is to specify the name of the algorithm, that we want to use

# Let's obtain a reference to the pca container image
# Note that all  models are named estimators
# You don't have to specify (hardcode) the region, get_image_uri will get the current region name using boto3.Session


from sagemaker.amazon.amazon_estimator import get_image_uri


container = get_image_uri(boto3.Session().region_name, 'pca')
```

Then we specify the role, instance_count, instance_type for training and set the hyperparameters. In this code, we mentioned feature_dim=11 and num_componenets=6, this means we reduced 11 features into 6 Principal Components. 
```bash
# We have passed in the container, the type of instance that we would like to use for training 
# output path and sagemaker session into the Estimator. 
# We can also specify how many instances we would like to use for training


pca = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.m5.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sagemaker_session)

# We can tune parameters like the number of features that we are passing in, mode of algorithm, mini batch size and number of pca components


pca.set_hyperparameters(feature_dim=11,
                        num_components=6,
                        subtract_mean=False,
                        algorithm_mode='regular',
                        mini_batch_size=100)


# Pass in the training data from S3 to train the pca model


pca.fit({'train': s3_train_data})

# Let's see the progress using cloudwatch logs
```
We'll get an output like this:
```bash
#metrics {"StartTime": 1767441876.7668202, "EndTime": 1767441876.7673469, "Dimensions": {"Algorithm": "PCA", "Host": "algo-1", "Operation": "training"}, "Metrics": {"setuptime": {"sum": 20.956039428710938, "count": 1, "min": 20.956039428710938, "max": 20.956039428710938}, "totaltime": {"sum": 1176.9766807556152, "count": 1, "min": 1176.9766807556152, "max": 1176.9766807556152}}}

2026-01-03 12:04:51 Uploading - Uploading generated training model
2026-01-03 12:04:51 Completed - Training job completed
Training seconds: 115
Billable seconds: 115
```


- Deploy the trained PCA model.
```bash
# Deploy the model to perform inference 

pca_reduction = pca.deploy(initial_instance_count = 1,
                                          instance_type = 'ml.m5.xlarge')
```
Deploying an Endpoint takes a bit of time and we'll get this "!" at the end.
<img width="612" height="82" alt="image" src="https://github.com/user-attachments/assets/87b3fa5e-2744-41f4-89f2-2e6e84745b0b" />

Then we'll override the data that will be sending to the model. Basically the deployed model predict the data to be text/csv format. 
```bash
# Content type overrides the data that will be passed to the deployed model, since the deployed model expects data in text/csv format.

# Serializer accepts a single argument, the input data, and returns a sequence of bytes in the specified content type

# Deserializer accepts two arguments, the result data and the response content type, and return a sequence of bytes in the specified content type.

# Reference: https://sagemaker.readthedocs.io/en/stable/predictors.html


# pca_reduction.content_type = 'text/csv'
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

pca_reduction.serializer = CSVSerializer()
pca_reduction.deserializer = JSONDeserializer()
```

Then we take the results and extract predictions out of it.
```bash
# make prediction on the test data

result = pca_reduction.predict(np.array(df_final))

result # results are in Json format
```

To Iterate through the results.
```bash
# Since the results are in Json format, we access the scores by iterating through the scores in the predictions
predictions = np.array([r['projection'] for r in result['projections']])

predictions

predictions.shape
```

To delete the endpoint.
```bash
# Delete the end-point

pca_reduction.delete_endpoint()
```



- Train & Evaluate XGBoost model on data after dimensionality reduction.
```bash
# Convert the array into dataframe in a way that target variable is set as the first column and is followed by feature columns
# This is because sagemaker built-in algorithm expects the data in this format

train_data = pd.DataFrame({'Target':df_target})
train_data

for i in range(predictions.shape[1]):
    train_data[i] = predictions[:,i]

train_data.head()
```

Divide the data into Training and Testing set.
```bash
train_data_size = int(0.9 * train_data.shape[0])
train_data_size

# shuffle the data in dataframe and then split the dataframe into train, test and validation sets.

import sklearn 

train_data = sklearn.utils.shuffle(train_data)
train, test, valid = train_data[:train_data_size], train_data[train_data_size:train_data_size + 3500], train_data[train_data_size + 3500:]

train.shape, test.shape,valid.shape

X_test, y_test = test.drop(columns = ['Target']), test['Target']


# save train_data and validation_data as csv files

train.to_csv('train.csv',header = False, index = False)
valid.to_csv('valid.csv',header = False, index = False)
```


