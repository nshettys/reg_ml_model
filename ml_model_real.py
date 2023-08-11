import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
#to make a scatter plot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import  r2_score
import pickle


ca_housing = fetch_california_housing()
#print(ca_housing)

#Preparing the dataset
dataset = pd.DataFrame(ca_housing.data , columns= ca_housing.feature_names)
#print(dataset)
#print(dataset.head()) # remember that the data will be showcasing only feature columsn and not target

dataset['Price'] = ca_housing.target

#print(dataset.head())

#print(dataset.info())

# Summarizing the stats of the data
#print(dataset.describe())
#Check the missing value very important
#print(dataset.isnull().sum())

#Exploratory data analysis

#explore co-relation  - how the independent values are co related with the target values
#print(dataset.corr())

#sns.pairplot(dataset)
#plt.show()

#analysing the co related features
plt.scatter(dataset['HouseAge'], dataset['Price'])
#plt.xlabel("HouseAge")
#plt.ylabel("Price")
#plt.show()

#see it in a regression line
#sns.regplot(x="AveBedrms", y= "Price",data=dataset)
#plt.show()

#linerity is important

#Independent and Dependent features

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

#print(X.head())
#print(y.head())

#Training and test dataset

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)
#print(X_train)

#Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#print(X_train)

#Model Training

reg = LinearRegression()
reg.fit(X_train,y_train)

#will be operatin in hyper plane

#Print the coefficients and intercepts

print(reg.coef_)
print(reg.intercept_)

#on which parameter has the model been trained
print(reg.get_params())


#prediction with test data

reg_predict = reg.predict(X_test)
print(reg_predict)

#scatter plot for prediction
plt.scatter(y_test,reg_predict)
#plt.show()

#residual plotting error from ytest and reg pred
residuals = y_test - reg_predict
print(residuals)

#plot the residuals - should get normal distribution
sns.displot(residuals, kind ="kde")
#plt.show()

#scatter plot with respect to prediction and residuals - should have uniform distribution
plt.scatter(reg_predict,residuals)
#plt.show()

#performance metrics

print(mean_absolute_error(y_test,reg_predict))
print(mean_squared_error(y_test,reg_predict))
print(np.sqrt(mean_squared_error(y_test,reg_predict)))

#Rsquared and adjusted R Sqaure - more it goes towards 100% the better it is . currently it shows 59 %
score = r2_score(y_test,reg_predict)
print('score' , score)

#Adjusted R2

#New Data Prediction

print(ca_housing.data[0].shape)
#This is in one dimension

#Lets convert it to two dimension
print(ca_housing.data[0].reshape(1,-1))

#standardiize and transformation of new data
scaler.transform(ca_housing.data[0].reshape(1,-1))

print(reg.predict(ca_housing.data[0].reshape(1,-1)))

#Pickling the Model for deployment
pickle.dump(reg,open('regmodel.pkl','wb'))
pickled_model = pickle.load(open('regmodel.pkl','rb'))
print('pickled prediction', pickled_model.predict(ca_housing.data[0].reshape(1,-1)))