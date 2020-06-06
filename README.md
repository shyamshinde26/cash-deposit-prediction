### Project Overview

 The objective is to predict the cash deposit amount for branches for the year 2016. We are given customer data having features like Year Deposits Variables, Establishment Date, and Acquired Date, etc. Using this data of previous years we have to predict the deposit amount for the year 2016. 


### Learnings from the project

 By completing this project we will learn:

- Label Encoding

- One hot Encoding

- Model Building


### Approach taken to solve the problem

 First, we did the data cleaning process. After cleaning the data we did the feature engineering to extract new feature. To apply the linear regression model we had to encode the categorical data using label encoder we encode the data. To get a better result we try decision tree regressor and XGBoost.


### Challenges faced

 The main challenge was to find the age of the deposits (age of the bank account). To calculate this we used the estimated date and acquired date. Getting the difference between them will give us the age of the account.


