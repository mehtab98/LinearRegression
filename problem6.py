import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

#This function simply adds labels and cleans the data 
def clean_car_data():
	columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name","mile_category"]
	df = pd.read_csv("auto-mpg.data", delimiter = "\\s+", names = columns)
	clean_df = df[df.horsepower != '?']
	reset_df = clean_df.reset_index(drop=True)
	return reset_df 

#this function takes data and splits the set into either 'low','medium', 'high', or 'very high' based on mpg 
#we do this categoriziation by takinng the quartile distribution 
def quartile_categorize():
	data = clean_car_data()
	categories = pd.qcut(x=data["mpg"], q=4, labels=['low','medium', 'high', 'very high'])
	data["mile_category"] = categories
	return data

def logistic_reg(): 
	data = quartile_categorize() 
	val = data 

	target = data["mile_category"]
	data = data.drop(['car_name','mpg', "mile_category"], axis=1)
	
	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)

	#model the logistic regression using lbfgs solver 
	model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter = 10000)
	
	
	model.fit(X_train, y_train)
	

	print("The precesion calculated for problem 6 LogisticRegression is: ")
	print(model.score(X_test, y_test))
	

	


logistic_reg()