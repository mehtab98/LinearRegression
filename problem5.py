import pandas as pd 
import numpy as np
from problem4 import *

global zero_weight
global one_weight 
global two_weight

share_var()

from problem4 import datar 

from problem4 import data_train  
from problem4 import data_test  

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

	total_rows = data.shape[0] 
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



#Problem also utilizes function single_feature_order which can be found right above
def polynomial_feature_solver(data, order):
	global secondth_weight 
	col_names = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]


	y = dataz_train["mpg"].reset_index(drop=True)

	#used to initalize a MSE df 
	stats = (4,2) 
	matrix = np.ones(shape = stats, dtype = float) 
	columnz = ["Train Polynomial Regression", "Test Polynomial Regression"] 
	cf = pd.DataFrame(data=matrix, columns = columnz, index = ['0th','1st','2nd','3rd'] ) 

	zero_weight = 0 
	one_weight = 0 
	two_weight = 0
	i = 0 

	#calculate mse for training 
	while i <= order: 	
		x = x_generator(dataz_train, i)
		#weight = (X_transpose * X)^-1 * X^T * Y
		weight = np.dot(np.dot(np.linalg.pinv(np.dot(x.T,x)),x.T),y)
		
		#mpg = x * weight 
		calculated_mpg = x.dot(weight)
		mse = mean_squared_error(y,calculated_mpg)  

		if(i == 0):
			cf["Train Polynomial Regression"]['0th'] = mse 
			zero_weight = weight
			
		elif(i == 1):
			cf["Train Polynomial Regression"]['1st'] = mse 

			one_weight = weight

		else: 
			cf["Train Polynomial Regression"]['2nd']= mse 
			two_weight = weight 

		i = i + 1 

	
	y = data_test["mpg"].reset_index(drop=True)

	

	#initalize DF to graph regression 
	matrix = np.ones(shape = (100,4), dtype = float)
	columnz = ["in", "0", "1","2"]
	plot_data = pd.DataFrame(data=matrix, columns = columnz) 
	plot_data['in'] = y

	#calculating mse for training data 
	i = 0 
	while i <= order: 	
		x = x_generator(dataz_test, i)

				  		
		if(i == 0):
			calculated_mpg = x.dot(zero_weight)
			plot_data['0'] = calculated_mpg

			mse = mean_squared_error(y,calculated_mpg)
			cf["Test Polynomial Regression"]['0th'] = mse 

		elif(i == 1):
			calculated_mpg = x.dot(one_weight)

			plot_data['1'] = calculated_mpg

			mse = mean_squared_error(y,calculated_mpg)
			cf["Test Polynomial Regression"]['1st'] = mse 
		
		else: 
			calculated_mpg = x.dot(two_weight)

			plot_data['2'] = calculated_mpg

			mse = mean_squared_error(y,calculated_mpg)

			cf["Test Polynomial Regression"]['2nd']= mse

		i = i + 1 
	print('\n')	
	print('Calculed MPG for problem 5 from order 0 to 2:')
	print(plot_data)
	print('\n')
	print('Calculed MSE for problem 5:')
	print(cf)

	secondth_weight = two_weight

	return cf


polynomial_feature_solver(datar, 2)


