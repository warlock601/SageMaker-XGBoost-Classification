# SageMaker XGBoost
- XGBoost or Extreme Gradient Boosting algorithm is one of the most famous and powerful algorithms to perform both regression and classification tasks. 
- XGBoost is a supervised learning algorithm and implements gradient boosted trees algorithm. 
- The algorithm work by combining an ensemble of predictions from several weak models.

<img width="640" height="270" alt="image" src="https://github.com/user-attachments/assets/439f1d8d-a1b6-44c2-8bf3-1a05c1cbde52" /> </br>
Combining a majority vote out of all different trees will generate a more robust prediction compared to only using one single tree. 

### Why does Xgboost work really well? 
- Since the technique is an ensemble algorithm, it is very robust and could work well with several data types and complex distributions. 
- Xgboost has a many tunable hyperparameters that could improve model fitting.

### What are the applications of XGBoost?
XGBoost could be used for fraud detection to detect the probability of a fraudulent transactions based on transaction features. 

### Ensemble Learning
Ensemble learning is a machine learning technique that combines multiple individual models (called "base learners") to create a single, stronger model with better predictive performance, accuracy, 
and robustness than any single model could achieve alone, working on the principle that a group's decision is better than one's. 

- Ensemble techniques such as bagging and boosting can offer an extremely powerful algorithm by combining a group of relatively weak/average ones. 
- For example, you can combine several decision trees to create a powerful random forest algorithm
- By Combining votes from a pool of experts, each will bring their own experience and background to solve the problem resulting in a better outcome.
- Bagging and Boosting can reduce variance and overfitting and increase the model robustness. 

### BAGGING
Bagging aims to reduce variance and prevent overfitting by training multiple models in parallel. It creates multiple random subsets of the original dataset (through random sampling with replacement, known as bootstrapping).

### BOOSTING
Boosting aims to reduce bias and convert weak learners into a single strong learner by training models sequentially. Models are built one after another, and each new model focuses on correcting the errors made by the previous ones. 
Initially, all data points have equal weight, but misclassified points are given higher weight in subsequent iterations.


Accuracy is generally misleading and is not enough to assess the performance of a classifier. That is why we look at other metrics such as Precision and Recall. Especially if the dataset is unbalanced.

### PRECISION
Precision is an important metric when False positives are important (how many times a model says a pedestrian was detected and there was nothing there!

𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 =(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆)/(𝑇𝑂𝑇𝐴𝐿 𝑇𝑅𝑈𝐸 𝑃𝑅𝐸𝐷𝐼𝐶𝑇𝐼𝑂𝑁𝑆)

𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 =(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆)/(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆+𝐹𝐴𝐿𝑆𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆)


### RECALL
Recall is also called True Positive rate or sensitivity. Important metric when we care about false negatives.

𝑅𝑒𝑐𝑎𝑙𝑙 =(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆)/(𝐴𝐶𝑇𝑈𝐴𝐿 𝑇𝑅𝑈𝐸 )

𝑅𝑒𝑐𝑎𝑙𝑙 =(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆)/(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆+𝐹𝐴𝐿𝑆𝐸 𝑁𝐸𝐺𝐴𝑇𝐼𝑉𝐸𝑆)

#### Consider this Bank Fraud Detection Confusion Matrix:

<img width="725" height="373" alt="image" src="https://github.com/user-attachments/assets/86b680f4-a66d-4a98-bfb3-9e032bb3ab69" />


#### Consider the case of Spam Email Detection:

<img width="725" height="373" alt="image" src="https://github.com/user-attachments/assets/34c1b136-6eed-4e84-8c64-f567f83d6ff7" />

Basically the motive of above 2 snaps is to understand that sometimes Precision matters more and sometimes Recall, depending upon the use-case.


### F1 Score
- F1 Score is an overall measure of a model's accuracy that combines precision and recall. F1 score is the harmonic mean of precision and recall.
- What is the difference between F1 Score and Accuracy? </br>
In unbalanced datasets, if we have large number of true negatives (healthy patients), accuracy could be misleading. Therefore, F1 score might be a better KPI
to use since it provides a balance between recall and precision in the presence of unbalanced datasets. 

𝐹1 𝑆𝑐𝑜𝑟𝑒 =(2 ∗(𝑃𝑅𝐸𝐶𝐼𝑆𝐼𝑂𝑁 ∗𝑅𝐸𝐶𝐴𝐿𝐿))/((𝑃𝑅𝐸𝐶𝐼𝑆𝐼𝑂𝑁+𝑅𝐸𝐶𝐴𝐿𝐿))

𝐹1 𝑆𝑐𝑜𝑟𝑒 =(2 ∗𝑇𝑃)/(2∗𝑇𝑃+𝐹𝑃+𝐹𝑁)

<img width="472" height="137" alt="image" src="https://github.com/user-attachments/assets/9b67aa46-0102-4a81-8d7b-7960622a4e5b" />


### ROC CURVE
- ROC Curve is a metric that assesses the model ability to distinguish between binary (0 or 1) classes. 
- The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. 
- The true-positive rate is also known as sensitivity, recall or probability of detection in machine learning. 
- The false-positive rate is also known as the probability of false alarm and can be calculated as (1 − specificity). 
- Points above the diagonal line represent good classification (better than random)
- The model performance improves if it becomes skewed towards the upper left corner. 
<img width="352" height="360" alt="image" src="https://github.com/user-attachments/assets/ac51a2d6-68c6-49ac-961d-449530cb0498" />


### AUC (Area Under Curve)
- The light blue area represents the area Under the Curve of the Receiver Operating Characteristic (AUROC). 
- The diagonal dashed red line represents the ROC curve of a random predictor with AUROC of 0.5. 
- If ROC AUC = 1, perfect classifier
- Predictor #1 is better than predictor #2
- Higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s. 
<img width="377" height="237" alt="image" src="https://github.com/user-attachments/assets/1f08fe5b-2690-47da-8e15-c85091179c42" />

