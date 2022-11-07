#NumPy is the Python library used for working with arrays and to perform a wide variety of mathematical operations on arrays.
import numpy as np
#Pandas is the Python library for data analysis , here we have used it for reading the dataset.
import pandas as pd
#The matplotlib library Matplotlib is a cross-platform, data visualization and graphical
# plotting library for Python and its numerical extension NumPy.# As such, it offers a viable open source alternative to MATLAB.
#
#pyplot is a collection of functions that make matplotlib work like MATLAB.
import matplotlib.pyplot as plt
# It is also a Python library used for plotting graphs with the help of Matplotlib, Pandas, and Numpy.
# It is built on the roof of Matplotlib and is considered as a superset of the Matplotlib library.
import seaborn as sns
#Python Scikit-learn lets users perform various Machine Learning tasks and provides a means to implement Machine Learning in Python.
import sklearn

#To check the current versions of the libraries present in the system
("numpy version: "+np.__version__)
print("pandas version: "+pd.__version__)
print("seaborn version: "+sns.__version__)
print("sklearn version: "+sklearn.__version__)

#reads the dataset from the location saved in the local system.
dataset = pd.read_csv('C:/Users/user/Downloads/50_Startups.csv')
print(dataset)

#returns the desciption of the data in dataframe.
dataset.describe()

#checking for dupicate values
dataset.duplicated().sum()

# checking for null values
dataset.isnull().sum()

#method prints information about the DataFrame. The information contains the number of columns, column labels,
# column data types, memory usage, range index, and the number of cells in each column (non-null values).
dataset.info()

# correlation matrix
c = dataset.corr()

#ustomize the colors in the heatmap with the parameter of the heatmap() function in seaborn.
sns.heatmap(c,annot=True,cmap='Blues')
plt.show()

#outliers detection in the targeted variable
outliers = ['Profit']
plt.rcParams['figure.figsize'] = [8,8]
sns.boxplot(data=dataset[outliers], orient="v", palette="Set2" , width=0.7)
plt.title("Outliers Variable Distribution")
plt.ylabel("Profit Range")
plt.xlabel("Continuous Variable")
plt.show()

dataset.plot(kind='box',subplots=True,layout=(2,2),figsize=(12,7))

# Histogram on Profit
sns.distplot(dataset['Profit'],bins=5,kde=True)
plt.show()

sns.pairplot(dataset)
plt.show()

# spliting Dataset in Dependent & Independent Variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#training and testing the dataset for the machine to learn
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=0)

#linear_model is a class of the sklearn module if contain different functions for performing machine learning with linear models.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train,y_train)
#The dataset model has been tested.
print('Model has been trained successfully')

#given a trained model, predict the label of a new set of data.
y_pred = model.predict(x_test)
y_pred

#Returns the coefficient of determination R^2 of the prediction of the testing data.
testing_data_model_score = model.score(x_test, y_test)
print("Model Score/Performance on Testing data",testing_data_model_score)

#Returns the coefficient of determination R^2 of the prediction of the training data.
training_data_model_score = model.score(x_train, y_train)
print("Model Score/Performance on Training data",training_data_model_score)

df = pd.DataFrame(data={'Predicted value':y_pred.flatten(),'Actual Value':y_test.flatten()})
print(df)

#r2_score is the coefficient of determination of regression score function
from sklearn.metrics import r2_score

r2Score = r2_score(y_pred, y_test)
print("R2 score of model is :" ,r2Score*100)

#The Mean Squared Error (MSE) of an estimator measures the average of error squares
# i.e. the average squared difference between the estimated values and true value.
from sklearn.metrics import mean_squared_error

#The Mean Squared Error (MSE)of an estimator measures the average of error squares
# i.e. the average squared difference between the estimated values and true value
mse = mean_squared_error(y_pred, y_test)
print("Mean Squarred Error is :" ,mse*100)

#RSME (Root mean square error) calculates the transformation between values predicted by a model and actual values.
# it is one such error in the technique of measuring the precision and error rate of any machine learning algorithm of a regression problem.
rmse = np.sqrt(mean_squared_error(y_pred, y_test))
print("Root Mean Squarred Error is : ",rmse*100)

#metrics package calculates the accuracy score for a set of predicted labels against the true labels.
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_pred,y_test)
print("Mean Absolute Error is :" ,mae)