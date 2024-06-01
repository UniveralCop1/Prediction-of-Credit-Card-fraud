# FindDefault (Prediction of Credit Card fraud)



# Problem Statement:
        A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card   fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
We have to build a classification model to predict whether a transaction is fraudulent or not.
In this project we want to identify fraudulent transactions with Credit Cards. Our objective is to build a Fraud detection system using Machine learning techniques. In the past, such systems were rule-based. Machine learning offers powerful new ways.

# Techniques used in the project
 - Importing Libraries: The code starts by importing the necessary libraries, including Pandas, Scikit-learn, SciPy, Matplotlib, and Seaborn.
 - Data Loading and Exploration : The code loads the 'Creditcard.csv' dataset into a Pandas DataFrame. It then explores the data by printing the shape, information, and the first few rows of the DataFrame.
 - Class Distribution Visualization: The code displays the class distribution of the target variable ('Class') using a countplot from Seaborn.
 - Summary Statistics: The code calculates the summary statistics (mean, standard deviation, minimum, maximum, etc.) of the DataFrame and displays the class distribution.
 - Seaborn Styling: The code sets the Seaborn style to 'whitegrid' for better visualization.
 - Correlation Matrix: The code generates a correlation matrix for the DataFrame and visualizes it using a heatmap from Seaborn.
 - Transaction Amount Distribution: The code plots the distribution of the 'Amount' feature using a histogram with a kernel density estimation (KDE) line.
 - Time vs. Amount Visualization: The code creates a scatter plot of 'Time' vs. 'Amount' feature, with the 'Class' variable represented by the color of the points.
 - Boxplots and Histograms for Selected Features: The code selects five features ('V1', 'V2', 'V3', 'V4', 'V5') and creates a grid of boxplots and histograms to visualize the distribution of these features, grouped by the 'Class' variable.

# models implemented in the project
 - Logistic Regression
 - Random Forest Classifier
 - Gradient Boosting Classifier

# Feature Engineering Steps
 - Handle Missing Values: The code assumes that there are no critical missing values and drops rows with any missing data.
 - Scale the Features: The features are scaled using StandardScaler to ensure they are on a similar scale.
 - Handle Outliers: The code identifies and removes outliers using the Z-score method.
 - Create Derived Features: A new feature, 'Amount_per_Time', is created by dividing the 'Amount' feature by the 'Time' feature.
 - Feature Selection: Mutual Information Classifier (MIC) is used to select the top 10 most important features.
 - Dimensionality Reduction: Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset.

# Model Selection 
  - Importing Required Libraries: The code imports necessary libraries for data 
    manipulation (Pandas), machine learning (Scikit-learn), statistical analysis (SciPy), 
     and data visualization (Matplotlib and Seaborn).
 -  Data Loading: The code loads the 'Creditcard.csv' dataset into a Pandas DataFrame df.
 -  Exploratory Data Analysis (EDA):
     o The code checks the shape of the dataset.
     o It displays the information about the dataset using df.info() and the first few rows 
        using df.head().
     o It calculates the class distribution and summary statistics of the dataset.
     o It creates several visualizations to understand the data, including:
	     .   Class distribution
	     .   Correlation matrix
	     .   Distribution of transaction amounts
	     .   Relationship between time and amount
	     .   Boxplots and histograms for selected features

![Screenshot 2024-05-31 114318](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/5c4e59b8-eaa3-4dc9-89b7-8de3f8bafde5)

![Screenshot 2024-05-31 114354](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/44f088f5-d6cc-43c6-ae63-515a9cc4322d)

![Screenshot 2024-05-31 114445](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/d852dbc4-820b-4db7-8be6-e78b6591f206)

![Screenshot 2024-05-31 114500](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/10113756-bf67-4cd0-87ae-e95f1410a379)

![Screenshot 2024-05-31 114515](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/cef522f4-5806-4ed2-987d-41301127a75e)

![Screenshot 2024-05-31 114530](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/c5f7c812-9b04-4656-bc8c-329d50776827)

![Screenshot 2024-05-31 114546](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/77fa9f77-d8a0-4aca-81b8-0f7d4fc17811)

![Screenshot 2024-05-31 114557](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/d4258640-71ea-46da-b518-4d5e690a915e)

![Screenshot 2024-05-31 114608](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/df1131d5-4070-40f3-9d29-c166708ee46b)

![Screenshot 2024-05-31 114619](https://github.com/UniveralCop1/Prediction-of-Credit-Card-fraud/assets/170419127/9d420d2e-2f07-4d02-bf94-48b1998509d1)
    
  - Model Selection: The code selects three different classification models for credit card 
    fraud detection:
     o	Logistic Regression (LogisticRegression)
     o	Random Forest Classifier (RandomForestClassifier)
     o	Gradient Boosting Classifier (GradientBoostingClassifier)
    
 These models are commonly used for binary classification tasks like fraud detection, and they 
 offer different trade-offs in terms of interpretability, flexibility, and performance.



# Model Interpretability

To improve the model interpretability and explore other classification models, we can make the following changes to the code:

        1. Implement Logistic Regression, Random Forest, and Gradient Boosting Classifiers
            
        2. Evaluate and Compare the Models

        3. Perform Hyperparameter Tuning

        4. Visualize Feature Importance
             
These changes will allows to:
1. Compare the performance of Logistic Regression, Random Forest, and Gradient Boosting 
   Classifiers with the SVM model.
2. Tune the hyperparameters of each model using RandomizedSearchCV to find the optimal 
   configurations.
3. Visualize the feature importance for the tree-based models (Random Forest and Gradient 
  Boosting), which can provide insights into the most important variables for the 
  classification task.

# Model Deployment

A model deployment plan outlines the steps necessary to take a machine learning model from development to production. It includes processes, tools, and best practices to ensure the model performs well in a real-world environment. 
Here is a comprehensive model deployment plan: 
1. Pre-Deployment Phase 
2. Deployment Phase
3. Post-Deployment Phase 


# Communication

   Here is a clear and comprehensive communication of the findings, methodology, and results.
Introduction Credit card fraud is a significant problem in the financial industry, causing substantial financial losses for both businesses and individuals. Detecting fraudulent transactions is crucial to mitigate these losses and protect consumers. In this analysis, we aim to develop a machine learning model that can effectively identify fraudulent credit card transactions
 - Methodology

`1. Exploratory Data Analysis (EDA)
 2. Data Balancing
 3. Model Training and Evaluation
   
- Results
  
 1. Model Performance: The final SVM model achieved an accuracy of 0.9960 on the test set,with a cross-validation score (mean) of 0.9956. The model also demonstrated high precision 
    (0.97), recall (1.00), and F1-score (0.98), indicating its effectiveness in detecting both 
    fraud and normal transactions.
 2. Confusion Matrix: The confusion matrix visually confirmed the model's strong performance, showing that the vast majority of fraud and normal transactions were correctly classified.

- Conclusion
  
  The results of this analysis demonstrate that the SVM model, with the optimal hyperparameters, is highly effective in detecting credit card fraud. This model can be a valuable tool for financial institutions 
  to enhance their fraud detection capabilities, ultimately reducing financial losses and protecting customers.


