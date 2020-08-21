import pandas as pd 
import numpy as np
from problem5 import * 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegressionCV 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
polynomial_feature_solver(datar, 2)

from problem5 import secondth_weight 

#This function is able to find a single-feature's matrix to the Nth order 
def single_feature_order(data, order, feature): 

	total_rows = data.shape[0]  # gives number of row count

	total_columns = order + 1


	stats = (total_rows, total_columns)
	feature = data[feature].reset_index(drop=True)
	
	current_col = 0 
	degree = 0 

	while current_col <= order:

		if(current_col == 0): 
			
			matrix = np.ones(shape = stats, dtype = float)

		elif(current_col == 1): 
			matrix[:, 1] = feature 
	
		else: 

			new_feature = np.power(feature.astype(float), current_col)
			matrix[:, current_col] = new_feature

		current_col = current_col + 1 
		degree = degree + 1



	return matrix

def x_generator(data, order): 

	col_names = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"] 

	total_rows = 1
	total_columns = (order * 7) + 1
	stats = (total_rows, total_columns)
	matrix = np.ones(shape = stats, dtype = float)
	

	feature = 0 
	current_col = 0
	while feature < 7:
		
		if(feature == 0):
			order_val = single_feature_order(data, order, 'cylinders')

		else: 

			order_val = single_feature_order(data, order, col_names[feature])

			order_val = np.delete(order_val, 0, axis = 1)

		
		i = 0 
		while(i < order_val.shape[1]):

			matrix[:,current_col] = order_val[:,i]
			current_col = current_col + 1
			i = i + 1 

		feature = feature + 1 


	return matrix

def calculate():
	
	data = {"cylinders": [4], "displacement": [400], "horsepower": [150], "weight": [3500], "acceleration":[8], 'model_year': [81], 'origin': [1]}
	data = pd.DataFrame.from_dict(data)  
	data_log = data 
	
	x = x_generator(data, 2)
	mpg = x.dot(secondth_weight)
	print('\n')
	print('Problem 8:')
	print('second-order, multi-variate polynomial regression expects mpg:')
	print(mpg[0])


	data = quartile_categorize() 
	val = data 

	

	target = data["mile_category"]
	data = data.drop(['car_name','mpg', "mile_category"], axis=1)
	#data_log = data_log.drop(['car_name','mpg', "mile_category"], axis=1)


	scaler = MinMaxScaler()
	data = scaler.fit_transform(data)
	

	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)
	model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter = 10000)

	model.fit(X_train, y_train)

	print('\n')
	print("The Category predicted is  ")
	print(model.predict(data_log)[0])


calculate()
