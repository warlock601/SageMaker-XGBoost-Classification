# SageMaker-XGBoost-Classification
This repository demonstrates the usage of Amazon SageMaker's XGBoost algorithm to perform classification tasks.


## Principal Component Analysis
PCA is an unsupervised machine learning algorithm that performs dimensionality reductions while attempting at keeping the original information unchanged. PCA works by trying to find a new set of features called components.
Components are composites of the uncorrelated given input features. </br>
In Amazon SageMaker PCA operates in two modes:
- Regular: works well with sparse data small (manageable) number of observations/features.
- Randomized: works well with large number of observations/features.

PCA takes the data points from the original data space and convert them into the component space. Basically it will form 2 components called PC1 (Principal Component 1) and PC2 (Principal Component 2). Let say we have data points in 3-D space, what PCA could do is take all these data points and simply convert them from the original data space to component space. Number of features is reduced and we kept the exact same amount of information (or there is a loss of little information). This is extremely helpful in improving the computational requirements needed by the algorithm. 
</br>
</br>
<img width="682" height="262" alt="image" src="https://github.com/user-attachments/assets/a90233a0-8c9b-4db0-b514-8c3841e2c55d" />

Let say we have data of customers of a bank, so we don't want to train the machine learning algorithm with the original features that includes all the information about your customer (age, salary, SSN) and all this info might be very critical and if the instance got hacked or any issue, then the customer info might get compromised. So what we can do instead is, take all the features, use PCA to encode that data and change them into Principal Components and so nobody will understand these not (Because it is a kind of encryption that happens)


## Input/Output Interface for the PCA Algorithm
- For training, PCA expects data provided in the train channel, and optionally supports a dataset passed to the test dataset, which is scored by the final algorithm. Both recordIO-wrapped-protobuf and CSV formats are supported for training. You can use either File mode or    Pipe mode to train models on data that is formatted as recordIO-wrapped-protobuf or as CSV.
- For inference, PCA supports text/csv, application/json, and application/x-recordio-protobuf. Results are returned in either application/json or application/x-recordio-protobuf format with a vector of "projections."

### EC2 Instance Recommendation for the PCA Algorithm
PCA supports CPU and GPU instances for training and inference. Which instance type is most performant depends heavily on the specifics of the input data. For GPU instances, PCA supports P2, P3, G4dn, and G5.


### Hyperparameters in PCA
<img width="1062" height="551" alt="image" src="https://github.com/user-attachments/assets/95cf3ad0-7e15-42d2-aaa8-f9e1b4c78d89" />
<img width="1061" height="332" alt="image" src="https://github.com/user-attachments/assets/046023b8-8e04-4f9b-b862-d05d772a2660" />

