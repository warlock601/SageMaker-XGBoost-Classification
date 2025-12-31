<img width="2112" height="126" alt="image" src="https://github.com/user-attachments/assets/fb2b36ab-de0a-4201-ba88-307a4777f142" /># SageMaker XGBoost
- XGBoost or Extreme Gradient Boosting algorithm is one of the most famous and powerful algorithms to perform both regression and classification tasks. 
- XGBoost is a supervised learning algorithm and implements gradient boosted trees algorithm. 
- The algorithm work by combining an ensemble of predictions from several weak models.

<img width="1040" height="370" alt="image" src="https://github.com/user-attachments/assets/439f1d8d-a1b6-44c2-8bf3-1a05c1cbde52" />
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


Accuracy is generally misleading and is not enough to assess the performance of a classifier. That is why we look at other metrics such as Precision and Recall.

### PRECISION
Precision is an important metric when False positives are important (how many times a model says a pedestrian was detected and there was nothing there!

𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 =(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆)/(𝑇𝑂𝑇𝐴𝐿 𝑇𝑅𝑈𝐸 𝑃𝑅𝐸𝐷𝐼𝐶𝑇𝐼𝑂𝑁𝑆)

𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛 =(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆)/(𝑇𝑅𝑈𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆+𝐹𝐴𝐿𝑆𝐸 𝑃𝑂𝑆𝐼𝑇𝐼𝑉𝐸𝑆)
