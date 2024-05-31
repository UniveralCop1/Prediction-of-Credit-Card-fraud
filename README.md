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




